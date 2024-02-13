import numpy as np
import pandas as pd

# Veri setini yükleme
data = pd.read_csv('spam_ham_dataset.csv', encoding='ISO-8859-1')
print(data.describe)

# Veriyi karıştırma
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Veriyi eğitim ve test setlerine ayırma oranı
train_ratio = 0.8
train_size = int(len(data) * train_ratio)

# Veriyi eğitim ve test setlerine böleme
train_data = data[:train_size]
test_data = data[train_size:]

# Ham ve spam mailleri eğitim setinde ayırma
train_ham_mails = train_data[train_data['label'] == 'ham']
train_spam_mails = train_data[train_data['label'] == 'spam']

# Ham ve spam mailleri test setinde ayırma
test_ham_mails = test_data[test_data['label'] == 'ham']
test_spam_mails = test_data[test_data['label'] == 'spam']

# Eğitim seti için ham ve spam maillerin olasılıklarını hesaplama
p_ham = len(train_ham_mails) / len(train_data)
p_spam = len(train_spam_mails) / len(train_data)

# Kelime sayılarını hesaplama
ham_words = ' '.join(train_ham_mails['text']).split()
spam_words = ' '.join(train_spam_mails['text']).split()

# Tüm kelimelerin toplam sayısı
total_words = len(set(ham_words + spam_words))

# Kelimelerin olasılıklarını hesaplama
word_counts_ham = {word: (ham_words.count(word) + 1) / (len(ham_words) + total_words) for word in set(ham_words)}
word_counts_spam = {word: (spam_words.count(word) + 1) / (len(spam_words) + total_words) for word in set(spam_words)}

# Örnek veri seti üzerinden tahmin yapma
def predict(text):
    words = text.split()
    p_h_given_text = np.log(p_ham)
    p_s_given_text = np.log(p_spam)
    for word in words:
        if word in word_counts_ham:
            p_h_given_text += np.log(word_counts_ham[word])
        else:
            p_h_given_text += np.log(1 / (len(ham_words) + total_words))
            
        if word in word_counts_spam:
            p_s_given_text += np.log(word_counts_spam[word])
        else:
            p_s_given_text += np.log(1 / (len(spam_words) + total_words))
    
    if p_h_given_text > p_s_given_text:
        return 'ham'
    else:
        return 'spam'

# Test seti üzerinde tahmin yapma
test_data['predicted'] = test_data['text'].apply(predict)

# Confusion Matrix hesaplama
confusion_matrix = pd.crosstab(test_data['label'], test_data['predicted'], rownames=['Actual'], colnames=['Predicted'])
print("\nConfusion Matrix:")
print(confusion_matrix)

# Accuracy, Precision, Recall ve F1 Score hesaplama
TP = confusion_matrix.loc['ham', 'ham']
FP = confusion_matrix.loc['spam', 'ham']
FN = confusion_matrix.loc['ham', 'spam']
TN = confusion_matrix.loc['spam', 'spam']

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * precision * recall / (precision + recall)

print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
