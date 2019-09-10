import nltk
import json
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import scrolledtext
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from os import path
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
from parse_pdf import *
from txt_processing import *


df, df_dwn, tokens, bag_of_words, bag_words_idf, doc_all, doc_all_tfidf = data(update=False)

wc = wrdcld(tokens)
wc.to_file("./pic/1.png")

# tkinter ----------------------------------------------------------
window = tk.Tk()
window.title("Papers with Code: the latest in Machine Learning")
window.geometry('900x700')

# Disable resizing the GUI
window.resizable(0, 0)

tab_control = ttk.Notebook(window)

# Frame -----------------------------------------
tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)
tab3 = ttk.Frame(tab_control)
tab_control.add(tab1, text='WordCloud: gives frequencies of tokens')
tab_control.add(tab2, text='Query: gives similarities')
tab_control.add(tab3, text='Papers & Codes: gives urls')

# WordCloud -------------------------------------
path_1 = "./pic/1.png"
img_1 = Image.open(path_1)
photo_1 = ImageTk.PhotoImage(img_1)
lbl1 = tk.Label(tab1, image=photo_1)
lbl1.pack()

# Query ------------------------------------------
# input que
que = tk.Entry(tab2, width=100)
que.pack()


def clicked():
    txt = que.get()
    q = query(txt, bag_words_idf, doc_all_tfidf)
    plot_query(q)

    path = "./pic/query.png"
    img = Image.open(path)
    img = img.resize((890, 260), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(img)
    lbl2.configure(image=photo)
    lbl2.image = photo

    for paper in q.keys():
        gitList.insert(tk.END, f'{paper}\n')
        for j in (df_dwn[df_dwn['pdf_title'] == str(paper)+'.pdf']['paper_pdf']):
            gitList.insert(tk.END, f"Paper: {j}\n")
        for i in df.iloc[list(chain.from_iterable(df_dwn[df_dwn['pdf_title'] == str(paper)+'.pdf']['github_index']))]['repo_url']:
            gitList.insert(tk.END, f'{i}\n')
        gitList.insert(
            tk.END, '\n---------------------------------------------------------------------------------------------\n')


btn = tk.Button(tab2, text='Search', command=clicked)
btn.pack()

path_2 = "./pic/query.png"
img_2 = Image.open(path_2)
img_2 = img_2.resize((890, 260), Image.ANTIALIAS)
photo_2 = ImageTk.PhotoImage(img_2)
lbl2 = tk.Label(tab2, image=photo_2)
lbl2.pack(fill='both')

# github address
scroll = tk.Scrollbar(tab3)
scroll.pack(side=tk.RIGHT, fill=tk.Y)
gitList = tk.Listbox(tab3, width=900, height=700, yscrollcommand=scroll.set)
gitList.pack(side=tk.LEFT)
scroll.config(command=gitList.yview)

tab_control.pack(expand=1, fill='both')
window.mainloop()
