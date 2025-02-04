# **Bigram Language Model and GPT from Scratch**

## **Why This Project?**
This project serves as an educational demonstration of how neural networks can be applied to language modeling. By implementing a simple **bigram language model** and a **GPT-style transformer**, I aim to illustrate:

- The **fundamental concepts** behind modern large language models (LLMs).
- How language models **learn statistical relationships** between words or characters.
- The **differences between basic and advanced models**, showcasing how simple methods scale up to more sophisticated architectures.
- How models **generate coherent text** by predicting the next token in a sequence.
- The step-by-step **engineering process** behind training neural networks for text generation.

This project is designed for those interested in **natural language processing (NLP)** and **deep learning**, providing a clear path from first principles to more advanced architectures.

---

## **Data Sets**
The model is trained on text from **The Wizard of Oz**, a well-known public domain book, as well as the **OpenWebTextCorpus**, a high-quality dataset of internet text.

- **OpenWebTextCorpus:** [https://skylion007.github.io/OpenWebTextCorpus/](https://skylion007.github.io/OpenWebTextCorpus/)
- **Wizard of Oz Text File:** A trimmed-down version of the book for training the bigram model.

Each dataset undergoes a different preprocessing approach:
- **Wizard of Oz:** Loaded as a single text file, tokenized at the character level.
- **OpenWebTextCorpus:** Extracted from `.xz` compressed files, preprocessed using memory mapping (`mmap`) for efficient loading.

---

## **Bigram Model**

The **bigram model** is a simple **character-level** language model that predicts the next character based on the previous one. Instead of explicitly computing conditional probabilities, the model uses an **embedding table (`nn.Embedding`)** to learn representations of character relationships.

### **1. Data Preparation**
- Downloads **The Wizard of Oz** text file as training data.
- Reads and preprocesses text, mapping characters to integers.
- Converts the text into a PyTorch tensor.
- Splits the dataset into **training and validation sets (80/20 split)**.

### **2. Model Definition**
- Implements a **bigram language model** using PyTorch.
- Uses an **embedding table** (`nn.Embedding`) to represent character relationships.
- Defines a **forward pass** to compute logits and loss.
- Implements a **generate** function for text prediction.

### **3. Training Setup**
- Uses the **Adam optimizer** for efficient gradient updates.
- Runs a **training loop**, updating model weights using backpropagation.
- Implements a **batching function** (`get_batch()`) that selects random text sequences, ensuring target sequences are offset by one character.
- Evaluates loss using **training and validation splits**.

### **4. Text Generation**
- Uses the trained model to **predict and generate new text** by iteratively sampling characters.

---

## **GPT Model**

The **GPT model** builds upon the bigram model by introducing **transformer-based improvements**, such as:

- **Self-attention mechanisms** to capture long-range dependencies.
- **Multi-head attention** for more expressive feature extraction.
- **Feedforward layers** for learning complex patterns.
- **Dropout and layer normalization** for improved training stability.

### **1. Data Preprocessing**
- Loads and processes text data from the **OpenWebTextCorpus**.
- Uses **memory mapping (`mmap`)** to efficiently read large text files without loading them entirely into memory.
- Splits the data into **training and validation sets**.
- Builds a **character vocabulary** for encoding.

### **2. Model Definition**
- Implements **transformer blocks**, including:
  - **Self-attention** for token interaction.
  - **Multi-head attention** for diversified feature learning.
  - **Feedforward layers** for non-linear transformations.
  - **Layer normalization and residual connections** for stability.

### **3. Training Process**
- Uses the **AdamW optimizer** for stability.
- Trains the model over **multiple iterations**, evaluating performance.
- Saves model checkpoints for later use.

### **4. Text Generation**
- Saves the trained model for future use.
- Generates text based on an input prompt by **iteratively sampling from token probabilities**.

---

## **Summary**
This project demonstrates how **language models are built from the ground up**. It starts with a **basic bigram model**, highlighting **statistical text generation**, and then progresses to a **GPT-like transformer** that can capture **long-term dependencies** in text.

By following this project, you can:
✅ Understand the **inner workings** of language models.  
✅ Learn about **training neural networks** for text generation.  
✅ See how **transformer-based architectures** improve over simpler models.  
✅ Experiment with **building your own LLMs** from scratch.

🔗 **Feel free to explore and build on this!** 🚀

