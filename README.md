# 🎓 Unibot: a university chatbot trained from scratch

A small neural network that understands what prospective students are asking (admissions, programs, tuition, scholarships, housing, campus life) and answers with the right reply.

No API, no big language model: just a few hundred example sentences, a bag-of-words, and a 3-layer network built with TensorFlow/Keras.

> Final-year project notebook (`lovablePFE.ipynb`), designed to run on Google Colab.

---

## What it does

You type a question, and the bot finds the **topic (intent)** of your message, then replies with one of the answers written for that topic.

```
You:     heyuuu
Chatbot: Welcome! How can I assist you today?

You:     i want to know your name
Chatbot: I'm EduBot, your bot assistant through your university adventure

You:     what is the capital of spain?
Chatbot: I'm sorry, I don't understand. Could you please rephrase your question?
```

- **34 topics** (intents), including admissions, tuition fees, scholarships and financial aid, accommodation, campus facilities, study abroad, e-learning, alumni network, academic calendar, research opportunities, and small talk (greetings, thanks).
- **Out-of-scope detection**: if the network isn't confident enough (below 40%), the bot says the question is outside university topics instead of guessing.
- **Bilingual touches**: the data includes some French words and topics, and the goodbye message is in French.

## How it works

```
 intents file  →  clean the text  →  bag of words  →  neural network  →  intent  →  reply
 (questions +     (tokenize,          (0/1 vector       (128 → 64 → N     (best       (random choice
  answers)         lemmatize)          per message)      softmax)          topic)      among answers)
```

1. **Collect**: every topic has example questions (`patterns`) and possible answers (`responses`).
2. **Preprocess**: split sentences into words (NLTK), drop words of 2 letters or fewer, lowercase, lemmatize. This builds the vocabulary (462 words in my data).
3. **Encode**: each sentence becomes a vector of 0s and 1s (is each vocabulary word present?). The label is one-hot encoded.
4. **Train**: a small Keras network, 200 epochs.
5. **Predict**: encode the user's message the same way and take the topic with the highest probability.
6. **Reply**: pick a random answer from that topic.

### The model

| Layer | Size | Notes |
|---|---|---|
| Input | vocabulary size | bag of words |
| Dense + ReLU | 128 | Dropout 0.5 |
| Dense + ReLU | 64 | Dropout 0.3 |
| Dense + Softmax | number of topics | one probability per topic |

About **70,000 parameters** (272 KB). Loss: categorical cross-entropy. Optimizer: Adam (learning rate 0.01).

Training accuracy reaches about **96%** after 200 epochs. This is measured on the training data itself, so it shows the network learned its examples, not how well it handles new questions.

## ⚠️ About the data

The chatbot depends entirely on one file, **`intentsEnglish2.json`**, which holds my university's questions and answers. **That file is not included in this repository**, so the notebook won't run as is.

To use the project you need to write your own intents file. The topics, answers and vocabulary all come from it, so a different university means a new file, but the code stays the same.

### Format

```json
{
  "intents": [
    {
      "tag": "tuition_fees",
      "patterns": [
        "How much is the tuition?",
        "What are the fees for international students?",
        "Is the university free?"
      ],
      "responses": [
        "Tuition depends on your program. You can find the full fee table on the admissions page."
      ]
    },
    {
      "tag": "salutation",
      "patterns": ["hello", "hi there", "good morning"],
      "responses": ["Welcome! How can I assist you today?"]
    }
  ]
}
```

**Tips for good data**
- Write **at least 5 to 10 different patterns per tag**, phrased the way real students write.
- Keep tags clearly different from each other, since similar tags confuse the network.
- Add a few off-topic sentences or small talk so the bot learns what *not* to answer.
- Use real answers (links, dates, contacts) from your own university.

## Run it

### On Google Colab (easiest)
1. Open `lovablePFE.ipynb` in [Google Colab](https://colab.research.google.com).
2. Upload your `intentsEnglish2.json` in the file panel on the left (or change the file name in the notebook).
3. Run all cells. The last cell starts the chat: type your question, and `quit` or `stop` to leave.

### On your computer
```bash
pip install tensorflow nltk numpy jupyter
jupyter notebook lovablePFE.ipynb
```
The first run downloads the NLTK data (`punkt`, `wordnet`) automatically.

## What's in the notebook

| Section | What happens |
|---|---|
| Imports | NLTK, NumPy, TensorFlow/Keras |
| Data preprocessing | Read the intents file, tokenize, lemmatize, build vocabulary and classes |
| Training data | Bag-of-words vectors and one-hot labels, shuffled |
| Model | Dense 128 → 64 → softmax, with dropout |
| Training | 200 epochs |
| Prediction functions | `clean_text`, `bag_of_words`, `pred_class`, `get_response` |
| Chat loop | Type a message, get a reply |

## Known limits

- **It only knows what's in the data.** A question that wasn't covered gets a refusal or a wrong topic.
- **Bag of words ignores word order and meaning**, so "not eligible" and "eligible" look similar to it.
- **Fixed answers**: it can't write new sentences or combine information, only pick a prepared reply.
- **No test set**: the accuracy shown is on training data, so I can't claim it generalizes well.
- **Confidence thresholds** (0.2 to list topics, 0.4 to answer) were set by hand, not tuned.

## Ideas to improve it

- Split the data into train and test sets to measure real accuracy.
- Replace bag of words with word embeddings or a pre-trained model for better understanding.
- Save the trained model and vocabulary, so the bot loads instantly instead of retraining.
- Put it behind a web interface (Streamlit or Flask).
- Move to retrieval-based answers (RAG) so replies come from real documents. That's the path I took in my other project, [RAG-Chatbot-2IS](https://github.com/ayaelmajjouti/RAG-Chatbot-2IS).

## Author

**Aya El Majjouti**
