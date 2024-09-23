import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def preprocess_text(text):
    return text.lower()

def calculate_tfidf(texts):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    return tfidf_matrix

def calculate_cosine_similarity(tfidf_matrix):
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return cosine_sim

def calculate_similarity_rate(cosine_sim):
    similarity_rate = (1 - cosine_sim) * 100  # 转换为百分比
    return similarity_rate

def write_output(output_path, similarity_rate):
    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write(f"{similarity_rate:.2f}%\n")  # 输出百分比格式

def main():
    if len(sys.argv) != 4:
        print("Usage: python main.py [original file] [plagiarized file] [output file]")
        return

    orig_path, plag_path, output_path = sys.argv[1:4]

    orig_text = read_file(orig_path)
    plag_text = read_file(plag_path)

    # 文本预处理
    orig_text = preprocess_text(orig_text)
    plag_text = preprocess_text(plag_text)

    # 构建TF-IDF矩阵
    tfidf_matrix = calculate_tfidf([orig_text, plag_text])

    # 计算余弦相似度
    cosine_sim = calculate_cosine_similarity(tfidf_matrix)

    # 计算重复率
    similarity_rate = calculate_similarity_rate(cosine_sim)

    # 输出结果
    write_output(output_path, similarity_rate)

if __name__ == "__main__":
    main()