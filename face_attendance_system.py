"""
Project: AI Real-Time Face Recognition & Automated Bootcamp Attendance System
Domain: Computer Vision & Educational Technology
Libraries: OpenCV (cv2.face LBPH), Pandas, NumPy, Matplotlib
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# CONSTANTS AND FILE PATHS
DATA_DIR = "attendance_data"
FACES_DIR = os.path.join(DATA_DIR, "registered_faces")
REGISTRY_FILE = os.path.join(DATA_DIR, "students_registry.csv")
ATTENDANCE_FILE = os.path.join(DATA_DIR, "bootcamp_attendance_log.csv")
MODEL_FILE = os.path.join(DATA_DIR, "trained_face_model.yml")
REPORT_IMAGE = "bootcamp_attendance_report.png"

# Ensure directories exist
os.makedirs(FACES_DIR, exist_ok=True)

# Load OpenCV built-in Haar Cascade Face Detector
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


# Helper function to initialize empty database files
def initialize_database():
    """Initializes empty CSV files for student registry and attendance records."""
    if not os.path.exists(REGISTRY_FILE):
        registry_df = pd.DataFrame(columns=["Numeric_ID", "Student_ID", "Student_Name", "Registration_Date"])
        registry_df.to_csv(REGISTRY_FILE, index=False)
        
    if not os.path.exists(ATTENDANCE_FILE):
        attendance_df = pd.DataFrame(columns=["Student_ID", "Student_Name", "Bootcamp_Day", "Date", "Time", "Status"])
        attendance_df.to_csv(ATTENDANCE_FILE, index=False)


# Step 1: One-time student face registration via webcam
def register_student(student_id, student_name, num_samples=25):
    """
    Captures facial image samples for a student via webcam,
    assigns a unique numeric ID, and saves face templates.
    """
    initialize_database()
    student_id = str(student_id).strip()
    student_name = str(student_name).strip()
    
    registry_df = pd.read_csv(REGISTRY_FILE, dtype=str)
    
    # Check if student is already registered
    if student_id in registry_df["Student_ID"].values:
        print(f"\n[!] Student '{student_name}' (ID: {student_id}) is already registered.")
        train_recognizer()
        return False
        
    numeric_id = len(registry_df) + 1
    student_folder = os.path.join(FACES_DIR, f"{numeric_id}_{student_id}")
    os.makedirs(student_folder, exist_ok=True)
    
    print(f"\n[+] Starting Registration for: {student_name} (ID: {student_id})")
    print("[+] Please look directly at the webcam. Capturing face samples...")
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    count = 0
    
    while cap.isOpened() and count < num_samples:
        success, frame = cap.read()
        if not success:
            print("[!] Could not access webcam.")
            break
            
        frame = cv2.flip(frame, 1)  # Mirror display
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
        
        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_img, (200, 200))
            
            # Save cropped face sample
            sample_path = os.path.join(student_folder, f"sample_{count}.jpg")
            cv2.imwrite(sample_path, face_resized)
            
            # Draw visual feedback
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Capturing: {count}/{num_samples}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
        cv2.putText(frame, f"Registering: {student_name} ({count}/{num_samples})", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Student Face Registration", frame)
        
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
    if count >= 10:
        new_entry = pd.DataFrame([{
            "Numeric_ID": str(numeric_id),
            "Student_ID": str(student_id),
            "Student_Name": str(student_name),
            "Registration_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }])
        registry_df = pd.concat([registry_df, new_entry], ignore_index=True)
        registry_df.to_csv(REGISTRY_FILE, index=False)
        print(f"[+] Registration successful! Saved {count} face samples for {student_name}.")
        train_recognizer()
        return True
    else:
        print("[!] Registration incomplete: Insufficient face samples captured.")
        return False


# Step 2: Train local LBPH face recognition model
def train_recognizer():
    """
    Trains the Local Binary Patterns Histograms (LBPH) Face Recognizer
    on all registered student face samples.
    """
    initialize_database()
    registry_df = pd.read_csv(REGISTRY_FILE, dtype=str)
    
    if not os.path.exists(FACES_DIR) or len(os.listdir(FACES_DIR)) == 0:
        print("[!] No registered face folders found to train.")
        return None
        
    faces = []
    labels = []
    
    # Iterate directly through all student face folders
    for folder_name in os.listdir(FACES_DIR):
        folder_path = os.path.join(FACES_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue
            
        try:
            numeric_id = int(folder_name.split("_")[0])
        except ValueError:
            continue
            
        for file in os.listdir(folder_path):
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(folder_path, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    faces.append(cv2.resize(img, (200, 200)))
                    labels.append(numeric_id)
                    
    if len(faces) == 0:
        print("[!] No face images found for training.")
        return None
        
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_FILE)
    print(f"[+] LBPH Face Recognizer successfully trained on {len(faces)} face samples across {len(set(labels))} registered students.")
    return recognizer


# Step 3: Live classroom face attendance scanner
def start_live_attendance(bootcamp_day="Day 04"):
    """
    Scans faces in real time from webcam, recognizes registered students,
    and logs attendance automatically without duplicate markings.
    """
    initialize_database()
    registry_df = pd.read_csv(REGISTRY_FILE, dtype=str)
    
    if not os.path.exists(MODEL_FILE) or registry_df.empty:
        print("\n[!] No trained model found. Attempting to train on existing face samples...")
        if train_recognizer() is None:
            print("[!] Please register at least one student first (Option 1).\n")
            return
            
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILE)
    
    # Mapping numeric ID to student info
    id_map = {int(row["Numeric_ID"]): (str(row["Student_ID"]), str(row["Student_Name"])) for _, row in registry_df.iterrows()}
    
    attendance_df = pd.read_csv(ATTENDANCE_FILE, dtype=str)
    today_date = datetime.now().strftime("%Y-%m-%d")
    
    # Keep track of students marked present today
    today_marked = set(attendance_df[attendance_df["Date"] == today_date]["Student_ID"].tolist()) if not attendance_df.empty else set()
    
    print(f"\nSTARTING LIVE FACE ATTENDANCE SCANNER ({bootcamp_day.upper()})")
    print("[+] Position your face in front of the camera.")
    print("[+] Press 'q' on the video window to stop the scanner.\n")
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("[!] Could not access webcam.")
            break
            
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(70, 70))
        
        for (x, y, w, h) in faces:
            face_roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            numeric_id, confidence = recognizer.predict(face_roi)
            
            # In LBPH, lower confidence value means a closer match (< 80 is confident)
            if confidence < 80 and numeric_id in id_map:
                student_id, student_name = id_map[numeric_id]
                match_pct = max(0, int(100 - confidence))
                
                # Check if already marked today
                if student_id not in today_marked:
                    now = datetime.now()
                    new_record = pd.DataFrame([{
                        "Student_ID": str(student_id),
                        "Student_Name": str(student_name),
                        "Bootcamp_Day": str(bootcamp_day),
                        "Date": str(today_date),
                        "Time": now.strftime("%H:%M:%S"),
                        "Status": "PRESENT"
                    }])
                    attendance_df = pd.concat([attendance_df, new_record], ignore_index=True)
                    attendance_df.to_csv(ATTENDANCE_FILE, index=False)
                    today_marked.add(student_id)
                    print(f"[*] Attendance Marked: {student_name} ({student_id}) at {now.strftime('%H:%M:%S')}")
                    
                # Visual Green Box for recognized student
                box_color = (0, 255, 0)
                status_text = f"{student_name} ({match_pct}%)"
                sub_text = "STATUS: PRESENT [MARKED]"
            else:
                # Visual Red Box for unregistered student
                box_color = (0, 0, 255)
                status_text = "Unknown Student"
                sub_text = "STATUS: UNREGISTERED"
                
            # Draw bounding box and text
            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
            cv2.rectangle(frame, (x, y-35), (x+w, y), box_color, cv2.FILLED)
            cv2.putText(frame, status_text, (x+6, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
            cv2.putText(frame, sub_text, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            
        # Top HUD Dashboard
        cv2.rectangle(frame, (0, 0), (640, 45), (30, 30, 30), cv2.FILLED)
        hud_text = f"Bootcamp Attendance | Present Today: {len(today_marked)}/{len(registry_df)} | Press 'q' to Quit"
        cv2.putText(frame, hud_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        
        cv2.imshow("AI Bootcamp Face Attendance Scanner", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[+] Scanner closed. Total students present today: {len(today_marked)}/{len(registry_df)}")


# Step 4: Visual attendance analytics and report generator
def generate_attendance_analytics():
    """
    Generates a 4-panel visual analytics report from real attendance records
    using Matplotlib and saves high-resolution chart.
    """
    initialize_database()
    if not os.path.exists(ATTENDANCE_FILE) or os.stat(ATTENDANCE_FILE).st_size == 0:
        print("\n[!] No attendance data available yet. Please record attendance first.\n")
        return
        
    df = pd.read_csv(ATTENDANCE_FILE, dtype=str)
    reg_df = pd.read_csv(REGISTRY_FILE, dtype=str)
    
    if df.empty:
        print("\n[!] Attendance log is currently empty.")
        print("[!] Register a student and run the Live Attendance Scanner to collect real data.\n")
        return
        
    print("\nCURRENT ATTENDANCE LOG:")
    print(df)
    
    fig = plt.figure(figsize=(15, 9))
    plt.suptitle("AI Bootcamp Attendance & Engagement Analytics Report", fontsize=16, fontweight='bold')
    
    # Subplot 1: Daily Attendance Count
    ax1 = fig.add_subplot(2, 2, 1)
    day_counts = df.groupby("Bootcamp_Day")["Student_ID"].nunique()
    bars = ax1.bar(day_counts.index, day_counts.values, color='royalblue', edgecolor='black', alpha=0.85)
    ax1.set_title("Attendance Count Across Bootcamp Days", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Bootcamp Day", fontweight='bold')
    ax1.set_ylabel("Number of Students Present", fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{int(yval)}", ha='center', va='bottom', fontweight='bold')
        
    # Subplot 2: Overall Attendance Rate (Pie Chart)
    ax2 = fig.add_subplot(2, 2, 2)
    total_possible = max(len(reg_df) * len(day_counts), len(df), 1)
    actual_attended = len(df)
    absent_count = max(0, total_possible - actual_attended)
    
    pie_values = [actual_attended, absent_count] if absent_count > 0 else [actual_attended]
    pie_labels = ["Present", "Absent"] if absent_count > 0 else ["Present (100%)"]
    pie_colors = ['#2ecc71', '#e74c3c'] if absent_count > 0 else ['#2ecc71']
    
    ax2.pie(pie_values, labels=pie_labels, autopct='%1.1f%%',
            colors=pie_colors, startangle=140, shadow=True)
    ax2.set_title("Overall Attendance Ratio (%)", fontsize=12, fontweight='bold')
    
    # Subplot 3: Student-Wise Attendance Frequency
    ax3 = fig.add_subplot(2, 2, 3)
    student_attendance = df["Student_Name"].value_counts()
    ax3.barh(student_attendance.index, student_attendance.values, color='teal', edgecolor='black', alpha=0.8)
    ax3.set_title("Student-wise Session Participation", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Total Days Attended", fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    # Subplot 4: Hourly Check-in Arrival Time Distribution
    ax4 = fig.add_subplot(2, 2, 4)
    df['CheckIn_Hour'] = df['Time'].apply(lambda t: int(t.split(':')[0]) if isinstance(t, str) and ':' in t else 10)
    ax4.hist(df['CheckIn_Hour'], bins=np.arange(8.5, 23.5, 1), color='darkorange', edgecolor='black', alpha=0.85)
    ax4.set_title("Check-In Time Distribution (Punctuality)", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Hour of Day (24-hr format)", fontweight='bold')
    ax4.set_ylabel("Check-In Count", fontweight='bold')
    ax4.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(REPORT_IMAGE, dpi=300)
    print(f"\n[+] Visual Analytics Report generated and saved to '{REPORT_IMAGE}'")
    print("[+] Displaying chart window (close the graph window to return to the menu)...")
    
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\n[+] Chart display closed.")


# Main interactive controller
def main():
    try:
        print("\nAI BOOTCAMP REAL-TIME FACE RECOGNITION & ATTENDANCE SYSTEM\n")
        
        initialize_database()
        reg_count = len(pd.read_csv(REGISTRY_FILE, dtype=str))
        att_count = len(pd.read_csv(ATTENDANCE_FILE, dtype=str))
        print(f"[Database Status] Registered Students: {reg_count} | Total Attendance Records: {att_count}")
        
        print("\nChoose an option:")
        print("1. Register a New Student (One-time Webcam Face Capture)")
        print("2. Start Live Face Attendance Scanner (Classroom Webcam)")
        print("3. Generate & View Visual Attendance Analytics Report")
        
        choice = input("\nEnter choice (1/2/3): ").strip()
        
        if choice == "1":
            s_id = input("Enter Student ID (e.g., BC-01): ").strip()
            s_name = input("Enter Student Full Name: ").strip()
            if s_id and s_name:
                register_student(s_id, s_name)
            else:
                print("[!] Invalid ID or Name.")
        elif choice == "2":
            day = input("Enter Bootcamp Day (e.g., Day 01 / Day 02 / Day 04): ").strip() or "Day 04"
            start_live_attendance(day)
        elif choice == "3":
            generate_attendance_analytics()
        else:
            print("[!] Invalid selection. Exiting.")
    except KeyboardInterrupt:
        print("\n\n[+] Program terminated cleanly.")


if __name__ == "__main__":
    main()
