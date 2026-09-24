from textblob import TextBlob


def analyze_sentiment():
  print("--- Real-Time Sentiment Analyzer ---")
  print("Type any sentence to analyze. Type 'exit' to quit.\n")

  while True:
    user_input = input("Enter Text: ")

    if user_input.lower().strip() == "exit":
      print("Exiting Analyzer.")
      break

    if not user_input.strip():
      continue

    # Process text using TextBlob
    blob = TextBlob(user_input)
    polarity = blob.sentiment.polarity  # Range: -1.0 (Negative) to +1.0 (Positive)
    subjectivity = (
        blob.sentiment.subjectivity
    )  # Range: 0.0 (Factual) to 1.0 (Opinion)

    # Interpret results
    if polarity > 0.1:
      sentiment_label = "POSITIVE 🙂"
    elif polarity < -0.1:
      sentiment_label = "NEGATIVE 🙁"
    else:
      sentiment_label = "NEUTRAL 😐"

    print(f" -> Sentiment: {sentiment_label}")
    print(
        f" -> Score: Polarity={polarity:.2f}, Subjectivity={subjectivity:.2f}\n"
    )


if __name__ == "__main__":
  analyze_sentiment()