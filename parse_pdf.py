import os
from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

import pandas as pd
import re
import requests
import txt_processing
import pickle


def mkdir(direc):
    try:
        os.mkdir(direc)
    except OSError:
        pass
    else:
        pass


def data(update=False):
    df = pd.read_json('./data/links-between-papers-and-code.json')
    df['pdf_title'] = df['paper_title'].apply(lambda x: re.sub(r'[^\w\s]', '-', x) + '.pdf')

    if update:

        df_g = df.groupby(['pdf_title', 'paper_url_pdf'])
        data = list(df_g.groups.keys())

        title, paper = [], []

        for i in range(len(data)):
            title.append(data[i][0])
            paper.append(data[i][1])

        # dataframe for update
        df_dwn = pd.DataFrame({'pdf_title': title, 'paper_pdf': paper,
                               'github_index': list(df_g.groups.values())})
        dwnpdf(df_dwn)
        df_dwn.to_json('./data/groupby_papers.json')

        pdfdir = 'D:/Papers_with_Codes/pdf/'
        txtdir = 'D:/Papers_with_Codes/txt/'
        mkdir(txtdir)
        convertMultiple(pdfdir, txtdir)

        title_words(data=df_dwn['pdf_title'])  # title words for filter in text preprocessing

        doc_all = txt_processing.doc_text(txtdir)
        tokens, bag_of_words = txt_processing.data_preprocessing(doc_all.values())
        bag_words_idf = {word: txt_processing.idf(word, doc_all.values()) for word in bag_of_words}
        doc_all_tfidf = {doc: txt_processing.compute_tfidf(
            doc_all[doc], bag_words_idf) for doc in doc_all.keys()}

    else:
        df_dwn = pd.read_json('./data/groupby_papers.json')
        with open('./data/bag_words_idf.pickle', 'rb') as handle:
            bag_words_idf = pickle.load(handle)
        with open('./data/doc_all_tfidf.pickle', 'rb') as handle:
            doc_all_tfidf = pickle.load(handle)
        with open('./data/doc_all.pickle', 'rb') as handle:
            doc_all = pickle.load(handle)
        with open('./data/bag_of_words.pickle', 'rb') as handle:
            bag_of_words = pickle.load(handle)
        with open('./data/tokens.pickle', 'rb') as handle:
            tokens = pickle.load(handle)

    return df, df_dwn, tokens, bag_of_words, bag_words_idf, doc_all, doc_all_tfidf


def dwnpdf(df):

    path = 'D:/Papers_with_Codes/pdf/'

    update = set(df.pdf_title).difference(set(os.listdir(path)))
    df_update = df[df['pdf_title'].isin(update)]

    for title, url in zip(df_update.pdf_title, df_update.paper_pdf):
        res = requests.get(url)
        with open(f"{path}{title}", 'wb') as f:  # 二進位制寫入
            for content in res.iter_content():
                f.write(content)


def convert(fname, pages=None):
    if not pages:
        pagenums = set()
    else:
        pagenums = set(pages)

    output = StringIO()
    manager = PDFResourceManager()
    converter = TextConverter(manager, output, codec='utf-8', laparams=LAParams())
    interpreter = PDFPageInterpreter(manager, converter)
    infile = open(fname, 'rb')
    for page in PDFPage.get_pages(infile, pagenums):
        interpreter.process_page(page)
    infile.close()
    converter.close()
    text = output.getvalue()
    output.close
    return text


def convertMultiple(pdfDir, txtDir):
    for pdf in os.listdir(pdfDir):  # iterate through pdfs in pdf directory
        fileExtension = pdf.split(".")[-1]
        if fileExtension == "pdf":
            pdfFilename = pdfDir + pdf
            txt = pdf[:-4] + ".txt"
            textFilename = txtDir + txt
            if txt not in os.listdir(txtDir):
                try:
                    text = convert(pdfFilename)
                    # get string of text content of pdf
                    textFile = open(textFilename, "w", encoding='utf-8')
                    # make text file
                    textFile.write(text)  # write text to text file
                    textFile.close()
                    print("finish convert to txt", pdf)
                except:
                    print('unexpected error while converting', pdf)
                    continue

    print("finish convert all file")
