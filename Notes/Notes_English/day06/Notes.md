# Notes

## 1 Image Classification Case

### 1.1 Introduction to the CIFAR10 Dataset

-  CIFAR is a dataset provided by the torchvision third-party package

- Training set: 50,000 samples  Test set: 10,000 samples

- Label y contains 10 categories → a 10-class classification problem

- Shape of one image: (32, 32, 3)

  ```python
  import torch
  import torch.nn as nn
  from torchvision.datasets import CIFAR10
  from torchvision.transforms import ToTensor
  import torch.optim as optim
  from torch.utils.data import DataLoader
  import time
  import matplotlib.pyplot as plt
  from torchsummary import summary
  
  # Number of samples per batch
  BATCH_SIZE = 8


  # todo: 1 - load the dataset and convert it into a tensor dataset
  def create_dataset():
    # root: directory path where the folder is located
    # train: whether to load the training set
    # ToTensor(): convert image data into tensor data
    train_dataset = CIFAR10(root='./data', train=True, transform=ToTensor())
    valid_dataset = CIFAR10(root='./data', train=False, transform=ToTensor())
    return train_dataset, valid_dataset


  # todo: 2 - build a convolutional neural network classification model
  # todo: 3 - model training
  # todo: 4 - model evaluation
  if __name__ == '__main__':
    train_dataset, valid_dataset = create_dataset()
    print('Mapping between image classes and labels ->', train_dataset.class_to_idx)
    print('train_dataset ->', train_dataset.data[0])
    print('train_dataset.data.shape ->', train_dataset.data.shape)
    print('valid_dataset.data.shape ->', valid_dataset.data.shape)
    print('train_dataset.targets ->', train_dataset.targets[0])

    # Display the image
    plt.figure(figsize=(2, 2))
    plt.imshow(train_dataset.data[1])
    plt.title(train_dataset.targets[1])
    plt.show()
  ```

### 1.2 Build a Classification Neural Network Model

```python
# todo: 2 - build a convolutional neural network classification model
class ImageModel(nn.Module):
    # todo: 2-1 build the init constructor to construct the neural network
    def __init__(self):
        super().__init__()
        # First convolutional layer
        # Input channels = 3, since one RGB image has 3 channels
        # Output channels = 6, meaning 6 neurons extract 6 feature maps
        # Convolution kernel size = 3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)

        # First pooling layer
        # Window size = 2*2
        # Stride = 2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)

        # Second pooling layer
        # Output feature map of the pooling layer: 16*6*6
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # First hidden layer
        # in_features: flatten the 16*6*6 3D matrix from the last pooling layer
        # into a 1D vector
        self.linear1 = nn.Linear(in_features=16 * 6 * 6, out_features=120)

        # Second hidden layer
        self.linear2 = nn.Linear(in_features=120, out_features=84)

        # Output layer
        # out_features = 10, since this is a 10-class classification problem
        self.out = nn.Linear(in_features=84, out_features=10)

    # todo: 2-2 build the forward function to implement forward propagation
    def forward(self, x):
        # First layer: convolution + activation + pooling
        x = self.pool1(torch.relu(self.conv1(x)))

        # Second layer: convolution + activation + pooling
        # x -> (8, 16, 6, 6), meaning 8 samples,
        # and each sample is 16*6*6
        x = self.pool2(torch.relu(self.conv2(x)))

        # First hidden layer can only receive 2D data
        # Convert the 4D tensor into a 2D tensor
        # x.shape[0]: number of samples in each batch
        # The last batch may contain fewer than 8 samples,
        # so we do not hardcode it as 8
        # -1 * 8 = 8 * 16 * 6 * 6
        # -1 = 16 * 6 * 6 = 576
        x = x.reshape(shape=(x.shape[0], -1))
        x = torch.relu(self.linear1(x))

        # Second hidden layer
        x = torch.relu(self.linear2(x))

        # Output layer
        # No softmax activation is used here,
        # because the multi-class cross-entropy loss function
        # will automatically apply softmax later
        x = self.out(x)
        return x

    # todo: 3 - model training


# todo: 4 - model evaluation
if __name__ == '__main__':
    train_dataset, valid_dataset = create_dataset()
    # print('Mapping between image classes and labels ->', train_dataset.class_to_idx)
    # print('train_dataset ->', train_dataset.data[0])
    # print('train_dataset.data.shape ->', train_dataset.data.shape)
    # print('valid_dataset.data.shape ->', valid_dataset.data.shape)
    # print('train_dataset.targets ->', train_dataset.targets[0])

    # # Display the image
    # plt.figure(figsize=(2, 2))
    # plt.imshow(train_dataset.data[1])
    # plt.title(train_dataset.targets[1])
    # plt.show()

    model = ImageModel()
    summary(model, input_size=(3, 32, 32))
```

### 1.3 Model Training

```python
# todo: 3 - model training
def train(train_dataset):

    # Create DataLoader
    dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Create model object
    model = ImageModel()

    # model.to(device='cuda')

    # Create loss function object
    criterion = nn.CrossEntropyLoss()

    # Create optimizer object
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Iterate through epochs
    # Define the number of epochs
    epoch = 10

    for epoch_idx in range(epoch):

        # Define total loss variable
        total_loss = 0.0

        # Define variable for counting correctly predicted samples
        total_correct = 0

        # Define variable for total number of samples
        total_samples = 0

        # Define start time variable
        start = time.time()

        # Iterate through the DataLoader (mini-batches)
        for x, y in dataloader:

            # print('y->', y)

            # Switch to training mode
            model.train()

            # Model predicts y
            output = model(x)

            # print('output->', output)

            # Compute loss value (average loss)
            loss = criterion(output, y)

            # print('loss->', loss)

            # Clear gradients
            optimizer.zero_grad()

            # Compute gradients
            loss.backward()

            # Update parameters
            optimizer.step()

            # Count the number of correctly predicted samples
            # tensor([9, 9, 9, 9, 9, 9, 9, 9])
            # print(torch.argmax(output, dim=-1))

            # tensor([False, False, False, False, False, False, False, False])
            # print(torch.argmax(output, dim=-1) == y)

            # tensor(0)
            # print((torch.argmax(output, dim=-1) == y).sum())
            total_correct += (torch.argmax(output, dim=-1) == y).sum()

            # Calculate the total loss for the current batch
            # loss.item(): average loss of the current batch
            total_loss += loss.item() * len(y)

            # Count the number of samples in the current batch
            total_samples += len(y)

        end = time.time()

        print('epoch:%2s loss:%.5f acc:%.2f time:%.2fs' % (
            epoch_idx + 1,
            total_loss / total_samples,
            total_correct / total_samples,
            end - start
        ))

    # Save the trained model
    torch.save(obj=model.state_dict(), f='model/imagemodel.pth')


# todo: 4 - model evaluation
if __name__ == '__main__':
    train_dataset, valid_dataset = create_dataset()

    # Model training
    train(train_dataset)
```

### 1.4 Model Evaluation

```python
# todo: 4 - model evaluation
def test(valid_dataset):

    # Create the test dataset DataLoader
    dataloader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create the model object and load the trained model parameters
    model = ImageModel()
    model.load_state_dict(torch.load('model/imagemodel.pth'))

    # Define variables to count the number of correctly predicted samples
    # and the total number of samples
    total_correct = 0
    total_samples = 0

    # Iterate through the DataLoader
    for x, y in dataloader:

        # Switch the model to inference mode
        model.eval()

        # Model prediction
        output = model(x)

        # Convert prediction scores into class labels
        y_pred = torch.argmax(output, dim=-1)

        print('y_pred->', y_pred)

        # Count the number of correctly predicted samples
        total_correct += (y_pred == y).sum()

        # Count the total number of samples
        total_samples += len(y)

    # Print accuracy
    print('Acc: %.2f' % (total_correct / total_samples))


if __name__ == '__main__':
    train_dataset, valid_dataset = create_dataset()

    # Model training
    # train(train_dataset)

    # Model prediction
    test(valid_dataset)
```

### Network Performance Optimization

- Methods to improve performance:

  - Increase the number of convolution kernels

  - Increase the number of neurons in fully connected layers

  - Reduce the learning rate

  - Add a Dropout layer

  ```python
  class ImageClassification(nn.Module):
    def __init__(self):
        super(ImageClassification, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, stride=1, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 128, stride=1, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.linear1 = nn.Linear(128 * 6 * 6, 2048)
        self.linear2 = nn.Linear(2048, 2048)

        self.out = nn.Linear(2048, 10)

        # Dropout layer, p represents the probability of neurons being dropped
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):

        x = torch.relu(self.conv1(x))
        x = self.pool1(x)

        x = torch.relu(self.conv2(x))
        x = self.pool2(x)

        # Since the last batch may contain fewer than 32 samples,
        # we flatten according to the actual batch size
        x = x.reshape(x.size(0), -1)

        x = torch.relu(self.linear1(x))

        # Dropout regularization
        # If training accuracy is much higher than test accuracy,
        # the model is suffering from overfitting
        x = self.dropout(x)

        x = torch.relu(self.linear2(x))
        x = self.dropout(x)

        return self.out(x)
  ```

## 2 Introduction to RNN

### 2.1 What is an RNN (Recurrent Neural Network)

- RNN is a neural network model used to process sequential data
- Sequential data
  - Data generated according to time steps
  - Previous data is related to later data

### 2.2 RNN Application Scenarios

- NLP
  - text generation
  - machine translation
- Speech translation
- Music generation

## 3 Word Embedding Layer

### 3.1 Purpose of the Embedding Layer

- Convert words into vector representations
- Low-dimensional dense vectors help models learn semantic relationships between words

### 3.2 Workflow of Word Embedding

- Use jieba or other tokenization tools to segment sentences
- Get the index of each word
- Convert indices into tensor objects
- Feed them into the embedding layer to obtain word vectors

### 3.3 Word Embedding Layer API Usage

```python
import torch
import jieba
import torch.nn as nn


def dm01():
    # One sentence contains multiple words
    text = '北京冬奥的进度条已经过半，不少外国运动员在完成自己的比赛后踏上归途。'

    # Use the jieba module for word segmentation
    words = jieba.lcut(text)

    # Return the list of words
    print('words->', words)

    # Create an embedding layer
    # num_embeddings: number of words
    # embedding_dim: dimension of the word vector
    embed = nn.Embedding(num_embeddings=len(words), embedding_dim=8)

    # Get the index of each word
    for i, word in enumerate(words):

        # Convert the word index into a tensor vector
        word_vec = embed(torch.tensor(data=i))

        print('word_vec->', word_vec)


if __name__ == '__main__':
	dm01()
```

## 4 Recurrent Network Layer

### 4.1 RNN Network Layer Principle

- Purpose: Process sequential text data.

- The input of each layer includes the hidden state from the previous time step and the current word vector.

- The output of each layer includes the current prediction score y1 and the hidden state of the current time step.
- Hidden state: Has the ability to store memory and capture contextual information.
- If there are multiple RNN layers (for generative AI tasks), only the hidden state of the last layer and the prediction output are used. The prediction output is then fed into a fully connected neural network to predict the next word.
  - If the vocabulary contains N words, the final layer predicts N classes. Each word corresponds to one class, and the word with the highest probability is selected as the prediction.
- h1 = relu(wh0+b+wx+b)
- y1 = wh1+b

### 4.2 Using the RNN Layer API

```python
import torch
import torch.nn as nn


def dm01():
    # Create an RNN layer
    # input_size: dimension of the word vector
    # hidden_size: dimension of the hidden state vector
    # num_layers: number of hidden layers, default is 1
    rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=1)

    # Input x
    # (5, 32, 128) -> (number of words in each sentence, number of sentences, word vector dimension)
    # For the RNN object, input_size = word vector dimension
    x = torch.randn(size=(5, 32, 128))

    # Hidden state h0 from the previous time step
    # (1, 32, 256) -> (number of hidden layers, number of sentences, hidden state vector dimension)
    # For the RNN object, hidden_size = hidden state vector dimension
    h0 = torch.randn(size=(1, 32, 256))

    # Call the RNN layer to output the current prediction values and the current hidden state h1
    output, h1 = rnn(x, h0)
    print(output.shape, h1.shape)


if __name__ == '__main__':
	dm01()
```

## 5 Text Generation Example

### 5.1 Build the Vocabulary

```python
import torch
import jieba
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time


# Load data, perform word segmentation, and build the vocabulary
def build_vocab():
    # Dataset path
    file_path = 'data/jaychou_lyrics.txt'

    # Storage for tokenization results
    # List of unique words
    unique_words = []

    # Tokenized words for each line of text
    # Replace text data with the result of a word list for each line
    # [[], [], []]
    all_words = []

    # Traverse each line of text in the dataset
    for line in open(file=file_path, mode='r', encoding='utf-8'):
        # Use the jieba module for word segmentation; the result is a list
        words = jieba.lcut(line)
        # print('words->', words)

        # Store all tokenization results in all_words, including duplicate words
        # [[], [], []]
        all_words.append(words)

        # Traverse the tokenization result and store unique words in unique_words
        # words -> ['想要', '有', '直升机', '\n']
        for word in words:
            # If the word is not in the vocabulary, then it is a new word
            if word not in unique_words:
                unique_words.append(word)

    # print('unique_words->', unique_words)
    # print('all_words->', all_words)

    # Number of words in the corpus
    word_count = len(unique_words)
    # print('word_count->', word_count)

    # Word-to-index mapping {word: word index}
    word_to_index = {}
    for idx, word in enumerate(unique_words):
        word_to_index[word] = idx
    # print('word_to_index->', word_to_index)

    # Represent the lyrics text using vocabulary indices
    corpus_idx = []
    # print('all_words->', all_words)

    # Traverse the tokenization results of each line
    # all_words -> [['老街坊', ' ', '小', '弄堂', '\n'], ['消失', '的', ' ', '旧', '时光', ' ', '一九四三', '\n']]
    for words in all_words:
        # Temporarily store the word indices of each line
        temp = []

        # Get each word in the line and its corresponding index
        # words -> ['老街坊', ' ', '小', '弄堂', '\n']
        for word in words:
            # Get the word index from the dictionary based on the word
            temp.append(word_to_index[word])

        # Add a space between the words of each line
        # This separates each line from the next one, since they have no semantic relationship
        temp.append(word_to_index[' '])

        # Get the index of each word in the current document
        # extend: split the elements of temp and store them in corpus_idx
        corpus_idx.extend(temp)

    return unique_words, word_to_index, word_count, corpus_idx


if __name__ == '__main__':
    unique_words, word_to_index, word_count, corpus_idx = build_vocab()
    # print('unique_words->', unique_words)
    # print('word_count->', word_count)
    # print('corpus_idx->', corpus_idx)
    # print('word_to_index->', word_to_index)
```

### 5.2 Build the Dataset Object

```python
# Build the dataset object
# Create a class that inherits from the base class torch.utils.data.Dataset
class LyricsDataset(torch.utils.data.Dataset):
    # Define the constructor
    # corpus_idx: lyrics text represented by vocabulary indices
    # num_chars: number of words in each sentence
    def __init__(self, corpus_idx, num_chars):
        self.corpus_idx = corpus_idx
        self.num_chars = num_chars

        # Count how many words are in the lyrics text, without deduplication
        self.word_count = len(corpus_idx)

        # Number of sentences that can be generated from the lyrics text
        self.number = self.word_count // self.num_chars

    # Override the magic method __len__, len() returns this value
    def __len__(self):
        return self.number

    # Override the magic method __getitem__
    # obj[idx] executes this method; iterating through the DataLoader also executes this method
    def __getitem__(self, idx):
        # Set the starting index start, which cannot exceed word_count - num_chars - 1
        # -1: because y is shifted one position forward based on x
        start = min(max(idx, 0), self.word_count - self.num_chars - 1)
        #
        end = start + self.num_chars

        # Get x
        x = self.corpus_idx[start: end]
        y = self.corpus_idx[start + 1: end + 1]
        return torch.tensor(x), torch.tensor(y)


if __name__ == '__main__':
    unique_words, word_to_index, word_count, corpus_idx = build_vocab()
    # print('unique_words->', unique_words)
    # print('word_count->', word_count)
    # print('corpus_idx->', corpus_idx)
    # print('word_to_index->', word_to_index)
    print('corpus_idx->', len(corpus_idx))
    dataset = LyricsDataset(corpus_idx, 5)
    print(len(dataset))

    # Get the first group of x and y
    x, y = dataset[49135]
    print('x->', x)
    print('y->', y)
```

### 5.3 Build the Network Model

```python
class TextGenerator(nn.Module):
    def __init__(self, unique_word_count):
        super(TextGenerator, self).__init__()

        # Initialize the embedding layer:
        # number of words in the corpus, word vector dimension = 128
        self.ebd = nn.Embedding(unique_word_count, 128)

        # Recurrent network layer:
        # word vector dimension = 128, hidden vector dimension = 256, number of layers = 1
        self.rnn = nn.RNN(128, 256, 1)

        # Output layer:
        # feature vector dimension 256 is the same as the hidden vector dimension,
        # output size = number of words in the vocabulary
        self.out = nn.Linear(256, unique_word_count)

    def forward(self, inputs, hidden):
        # Output dimension: (batch, seq_len, word vector dimension 128)
        # batch: number of sentences
        # seq_len: sentence length, i.e. how many words each sentence contains
        embed = self.ebd(inputs)

        # The representation of x for the RNN layer is (seq_len, batch, word vector dimension 128)
        # The representation of output is similar to input x: (seq_len, batch, hidden vector dimension 256)
        # The shape of hidden before and after must be the same,
        # so the batch size of the DataLoader should be fixed
        output, hidden = self.rnn(embed.transpose(0, 1), hidden)

        # The fully connected layer takes 2D data as input: number of words * word vector dimension
        # Input dimension: (seq_len * batch, hidden vector dimension 256)
        # Output dimension: (seq_len * batch, number of words in the corpus)
        # output: score distribution of each word, which will later be combined with softmax
        # to produce a probability distribution
        # output.shape[-1]: represents the word vector dimension
        output = self.out(output.reshape(shape=(-1, output.shape[-1])))

        # Return the network output
        return output, hidden

    def init_hidden(self, bs):
        # Initialization of the hidden layer:
        # [number of network layers, batch, hidden vector dimension]
        return torch.zeros(1, bs, 256)

if __name__ == "__main__":
    # Load data
    unique_words, word_to_index, unique_word_count, corpus_idx = build_vocab()
    model = TextGenerator(unique_word_count)
    for named, parameter in model.named_parameters():
        print(named)
        print(parameter)
```

### 5.4 Build the Training Function

```python
def train():
    # Build the vocabulary
    unique_words, word_to_index, unique_word_count, corpus_idx = build_vocab()

    # Dataset object LyricsDataset, with the __getitem__ method implemented
    lyrics = LyricsDataset(corpus_idx=corpus_idx, num_chars=32)

    # Check the number of sentences
    # print(lyrics.number)

    # Initialize the model
    model = TextGenerator(unique_word_count)

    # DataLoader object, passing the lyrics dataset object into it
    lyrics_dataloader = DataLoader(lyrics, shuffle=True, batch_size=5)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimization method
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Number of training epochs
    epoch = 10
    for epoch_idx in range(epoch):
        # Training time
        start = time.time()

        iter_num = 0  # number of iterations

        # Training loss
        total_loss = 0.0

        # Traverse the dataset
        # DataLoader will call dataset.__getitem__(index) in the background
        # to get the data and labels for each sample, and then combine them into a batch
        for x, y in lyrics_dataloader:
            print('y.shape->', y.shape)

            # Initialize the hidden state
            hidden = model.init_hidden(bs=5)

            # Model computation
            output, hidden = model(x, hidden)
            print('output.shape->', output.shape)

            # Compute the loss
            # y has shape (batch, seq_len), and needs to be converted into a 1D vector
            # -> indices of 160 words
            # output has shape (seq_len, batch, word vector dimension)
            # We first transpose y (to make it consistent with output) and then reshape it
            y = torch.transpose(y, 0, 1).reshape(shape=(-1,))
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1  # increment the iteration count
            total_loss += loss.item()

        # Print training information
        print('epoch %3s loss: %.5f time %.2f' % (epoch_idx + 1, total_loss / iter_num, time.time() - start))

    # Save the model
    torch.save(model.state_dict(), 'model/lyrics_model_%d.pth' % epoch)

if __name__ == "__main__":
    train()
```

### 5.5 Build the Prediction Function


```python
def predict(start_word, sentence_length):
    # Build the vocabulary
    unique_words, word_to_index, unique_word_count, _ = build_vocab()

    # Build the model
    model = TextGenerator(unique_word_count)

    # Load parameters
    model.load_state_dict(torch.load('model/lyrics_model_10.pth'))

    # Hidden state
    hidden = model.init_hidden(bs=1)

    # Convert the starting word into an index
    word_idx = word_to_index[start_word]

    # Storage location for generated word indices
    generate_sentence = [word_idx]

    # Traverse to the target sentence length and get each word
    for _ in range(sentence_length):
        # Model prediction
        output, hidden = model(torch.tensor([[word_idx]]), hidden)

        # Get the prediction result
        word_idx = torch.argmax(output)
        generate_sentence.append(word_idx)

    # Get the corresponding words according to the generated indices and print them
    for idx in generate_sentence:
        print(unique_words[idx], end='')


if __name__ == '__main__':
    # Call the prediction function
    predict('分手', 50)
```

