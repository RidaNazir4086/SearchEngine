from collections import OrderedDict


fRJ = open("relevance judgements.qrel", "r")
fscore = open("Dirichlet.txt", "r")
#fscore = open("OkapiBM25.txt", "r")
ranks_dic = {}
relevance ={}
query_rel_doc ={}

# dictinary for relevance judgment
prevQ = 0
for x in fRJ:
    qid,zero,doc,rel = x.split(" ")
    if int(prevQ) != int(qid.rstrip()):
        relevance[qid.rstrip()] = {'doc_rel': {}}
    relevance[qid.rstrip()]['doc_rel'][doc.rstrip()] = rel.rstrip()
    prevQ = int(qid.rstrip())

# dictionary for scoring file
prevQ = 0
for x in fscore:
    qid, doc,rank, s, r = x.split(" ")
    if int(prevQ) != int(qid.rstrip()):
        ranks_dic[qid.rstrip()] = {'doc_rank': {}}
    ranks_dic[qid.rstrip()]['doc_rank'][doc.rstrip()] = rank.rstrip()
    prevQ = int(qid.rstrip())

for x in relevance:
    query_rel_doc[x] =0
    for key,value in relevance[str(x)]['doc_rel'].items():
        if int(relevance[str(x)]['doc_rel'][key]) > 0:
            query_rel_doc[x] +=1


def P_at(k,qid):
    p=0
    for d,r in ranks_dic[str(qid)]["doc_rank"].items():
        if(int(r)==k):
            break
        if(d in relevance[str(qid)]['doc_rel']):
            if int(relevance[str(qid)]['doc_rel'][d]) > 0:
                p += 1
    return p/k



def MAP():
    avgP = 0
    for qid in ranks_dic.keys():
        relvantDoc =0
        rank = 0
        sum =0
        for d, r in ranks_dic[str(qid)]["doc_rank"].items():
            rank +=1
            if (d in relevance[str(qid)]['doc_rel']):
                if int(relevance[str(qid)]['doc_rel'][d]) > 0:
                    sum += P_at(rank,qid)
                    relvantDoc += 1
        #avgP += sum/relvantDoc
        avgP += sum /query_rel_doc[str(qid)]
    map =avgP/ len(ranks_dic)
    return map





# main

k=5
print("p@"+str(k)+" :")
for q in ranks_dic.keys():
    print("For Query "+q+": "+ str(P_at(k,int(q))))

print("********************************************")
k=10
print("p@"+str(k)+" :")
for q in ranks_dic.keys():
    print("For Query "+q+": "+ str(P_at(k,int(q))))

print("********************************************")
k=20
print("p@"+str(k)+" :")
for q in ranks_dic.keys():
    print("For Query "+q+": "+ str(P_at(k,int(q))))

print("********************************************")
k=30
print("p@"+str(k)+" :")
for q in ranks_dic.keys():
    print("For Query "+q+": "+ str(P_at(k,int(q))))

print("********************************************")
print("Mean Avg Precision: "+ str(MAP()))


#print(query_rel_doc)