import sys, json,random
with open('jsons.txt') as f:
    articles = f.readlines()
random.shuffle(articles)
articles = articles[:int(len(articles)*0.2)]
with open('french_news.txt','w') as f:
    for i,article in enumerate(articles):
        with open(article.strip()) as a:
            art = json.load(a)
            f.write(art['text']+'\n')
        print('{}/{}'.format(i,len(articles)))

