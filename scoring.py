import xml.etree.ElementTree as et
import sys
import os
import io , re
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
import math
import numpy as np

#flen = open("docLens.txt","w+")
f = open("stoplist.txt", "r")
stopWords = f.read()
fdoc= open("docids.txt", "r")
#docids = fdoc.read()
fTermIndex =open("term_index.txt", 'r')
invertedIndex = {}
fTerm = open("termids.txt" , 'r')

termIds={}
term_Ids={}
queryDic = {}
docID_len_DIc = {}
queryList = []
#Score = {} # sorted dictionary
Score_dic ={} # before sorting
doc_name ={}


mu = 2092.425
CorpusLen = 7313027
k1 = 1.2
k2 = 100
b = 0.75
D = 3495

def findingDocLens(fileName, dId=0):
    '''with os.scandir(fileName) as entries:
        for entry in entries:
            if entry.is_file():
                dId += 1
                with io.open("corpus/corpus/" + entry.name, 'r', encoding='utf8', errors='ignore') as f:
                    text = f.read()
                stemmed_Words = []
                soup = BeautifulSoup(text, "html.parser")
                cleantext = soup.get_text()  # parsing the html tags
                tokens = re.findall(r"\b([a-zA-Z]+)\b", cleantext)
                tokens = [token.lower() for token in tokens]
                filtered_tokens = [w for w in tokens if not w in stopWords]  # applying stop words
                # Applying stemming
                ps = PorterStemmer()
                for w in filtered_tokens:
                    stemmed_Words.append(ps.stem(w))

                flen.write(str(dId)+'\t'+ str(len(stemmed_Words))+'\n')
                docID_len_DIc[dId] = len(stemmed_Words)
    '''
    # redaing file for generating dictionary for documentid and length
    flen = open("docLens.txt", "r")
    for t in flen:
        dId, len = t.split("\t")
        docID_len_DIc[dId] = len.rstrip()

def termIndexProcessing():
    #for reading term_index file, decoding it and writing in decodedTerms.txt file
    '''fdecode = open("decodedTerms.txt", "w+")
    for t in fTerm:
        tId, term = t.split("\t")
        termIds[tId.rstrip()] = term

    lines = fTermIndex.readlines()
    for x in lines:
        word = (termIds[(x.split(" ")[0])]).rstrip()
        if (word in queryList):
            #print(word)
            invertedIndex[x.split(" ")[0]] = {'corpus_occurances': x.split(" ")[1], 'total_docs': x.split(" ")[2], 'tf': {}}
            fdecode.write(x.split(" ")[0] +" "+ x.split(" ")[1] +" "+ x.split(" ")[2])
            i=3
            decode = 0
            while(x.split(" ")[i]!="\n"):
                y=x.split(" ")[i]
                y=int(y.split(',')[0])
                decode= decode + y
                if(y != 0):
                    count=1
                    invertedIndex[x.split(" ")[0]]['tf'][str(decode)] = str(count)
                elif (y == 0):
                    count+=1
                    invertedIndex[x.split(" ")[0]]['tf'][str(decode)] = str(count)
                i+= 1
            for k,v in invertedIndex[x.split(" ")[0]]['tf'].items():
                fdecode.write(" "+k +","+ v)
            fdecode.write("\n") '''

    # for reading the decoded term file and storing it in dictionary
    fdecode = open("decodedTerms.txt","r")
    lines = fdecode.readlines()
    for x in lines:
        invertedIndex[x.split(" ")[0]] = {'corpus_occurances': x.split(" ")[1], 'total_docs': x.split(" ")[2], 'tf': {}}
        i=3
        count=0
        while count < int(x.split(" ")[2]):
            y = x.split(" ")[i]
            invertedIndex[x.split(" ")[0]]['tf'][(y.split(',')[0])] = (y.split(',')[1].rstrip())
            i+=1
            count+=1


def queryParsing(fileName):
    xml_path = fileName
    tree = et.parse(xml_path)
    root = tree.getroot()
    for id in root:
        x=id.attrib["number"]
        for q in id.findall('query'):
            queryDic[int(x)]= queryProcessing(q.text, stopWords)


def queryProcessing(query,stopWords):
    stemmed_Words = []
    #tokens= query.split()
    tokens = re.findall(r"\b([a-zA-Z]+)\b", query)
    #tokens = [token.lower() for token in tokens]
    filtered_tokens = [w for w in tokens if not w in stopWords]  # applying stop words
    # Applying stemming
    ps = PorterStemmer()
    for w in filtered_tokens:
         stemmed_Words.append(ps.stem(w))
    return stemmed_Words


def calculatin_mu():
    CorpusLen = 0
    for id,l in docID_len_DIc.items():
        CorpusLen+=int(l)
    #(len(docID_len_DIc))
    mu= CorpusLen / len(docID_len_DIc)
    print("mu: "+ str(mu))
    print("Corpus: " + str(CorpusLen))
    return mu, CorpusLen

def Dirichlet_Smoothing(q_id,query):
    for word in query:
        tid=term_Ids[word]
        for doc in invertedIndex[tid]["tf"]:
            lemda = int(docID_len_DIc[doc]) / (int(docID_len_DIc[doc]) + mu)
            score = math.log10(lemda *(int(invertedIndex[tid]["tf"][doc])/ int(docID_len_DIc[doc])) + ((1-lemda)*(int(invertedIndex[tid]['corpus_occurances'])/CorpusLen)))
            if not doc in Score_dic[q_id]["doc_score"]:
                Score_dic[q_id]["doc_score"][doc] = score
            else:
                Score_dic[q_id]["doc_score"][doc] += score

    Score = sorted(((value,key)  for (key,value) in Score_dic[q_id]["doc_score"].items()), reverse=True)
    #print(Score)
    return Score


def okapi_BM25(q_id, query):
    for word in query:
        tid = term_Ids[word]
        for doc in invertedIndex[tid]["tf"]:
            K = k1 * ((1 - b) + (b * (int(docID_len_DIc[doc]) / mu)))
            s1 = math.log2( (D+0.5) / (int(invertedIndex[tid]["total_docs"])+0.5))
            #s1 = np.log((D + 0.5) / (int(invertedIndex[tid]["total_docs"]) + 0.5))
            s2 = ((1+k1) * int(invertedIndex[tid]["tf"][doc])/ (K +int(invertedIndex[tid]["tf"][doc])))
            s3 = (((1+k2) * query.count(word))/ (k2 +query.count(word)))
            score= (s1*s2*s3)
            if not doc in Score_dic[q_id]["doc_score"]:
                Score_dic[q_id]["doc_score"][doc] = score
            else:
                Score_dic[q_id]["doc_score"][doc] += score

        Score = sorted(((value, key) for (key, value) in Score_dic[q_id]["doc_score"].items()), reverse=True)
        # print(Score)
        return Score


def printingScore(q_id,Score):
    rank=0
    prev=0
    for scores in Score:  # scores[0] = score, scores[1] = docid
        if prev != float(scores[0]):
            rank +=1
        #print(str(q_id) +" "+ str(scores[1]) +" "+str(rank)+" "+ str(scores[0]) +" "+ "run1"+"\n")
        fDirichlet.write(str(q_id) + " " + doc_name[str(scores[1])] + " " + str(rank) + " " + str(scores[0]) + " " + "run1" + "\n")
        #fOkapiBM.write(str(q_id) + " " + doc_name[str(scores[1])] + " " + str(rank) + " " + str(scores[0]) + " " + "run1" + "\n")
        prev = float(scores[0])




# main function
queryParsing('topics.xml')
for q in queryDic.values():
    for terms in q:
        queryList.append(terms)
termIndexProcessing()
findingDocLens("corpus/corpus/", 0)
for t in fTerm:
    tId, term = t.split("\t")
    term_Ids[term.rstrip()] = tId.rstrip()

for d in fdoc:
    dId, name = d.split("\t")
    doc_name[dId] = name.rstrip()

if sys.argv[1] == "BM25":
    fOkapiBM = open("OkapiBM25.txt", 'w+')

    for qid,q in queryDic.items():
        Score_dic[qid]={'doc_score' : {}}
        Score = okapi_BM25(qid,q)
        printingScore(qid,Score)
elif sys.argv[1] == "Dirichlet":
    fDirichlet = open("Dirichlet.txt", 'w+')

    for qid, q in queryDic.items():
        Score_dic[qid] = {'doc_score': {}}
        Score = Dirichlet_Smoothing(qid,q)
        printingScore(qid, Score)


