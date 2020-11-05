import csv
import sys
from collections import defaultdict
import numpy as np

csv_file = sys.argv[1]
header=None
allRows=[]
totalCorrect=0 #^666 66666666666666666666666666666666 666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666 666n6 66666 66 6 6 6 6 6 66  6 66  6 6b6666666666666666666666666    6 n6 6 6 66 6 6 6 6 66 66 6 6 66 6 6666   
totalGuessRight=0
noGoldGuessRight=0
goldRight=0
totalGold=0
#workerStats=defaultdict(lambda : {'guessRight': 0, 'noGoldGuessRIght':0, 'total': 0, 'correct': 0, 'goldRight':0, 'goldTotal':0, 'eGen':0, 'eReal':0})
workerStats=defaultdict(lambda : defaultdict(lambda:0))
total=0
annom=0

with open(csv_file) as f:
    reader = csv.reader(f, delimiter=',', quotechar='"')

    for row in reader:
        if header is None:
            header=row
            continue

        

        human = row[33]=='true'
        generated = row[32]=='true'
        #assert(human != generated)

        gt_human = row[31]=='real'
        gold_std = row[31]=='gold'

        if len(row[34])>0:
            correct =  row[34]=='false'  
        else:
            correct =  row[35]=='true'

        workerId = row[15]

        total+=1
        approve=True
        failedGold=False
        failedTrans=False
        workerStats[workerId]['total']+=1
        if gt_human==human and human!=generated:
            totalGuessRight+=1
            workerStats[workerId]['guessRight']+=1
            if not gold_std:
                noGoldGuessRight+=1
                workerStats[workerId]['noGoldGuessRight']+=1
                if human:
                    workerStats[workerId]['noGoldRightGuessHuman']+=1
                else:
                    workerStats[workerId]['noGoldRightGuessGen']+=1
        elif gt_human and not human and not gold_std:
            workerStats[workerId]['noGoldWrongGuessGen']+=1
        elif not gt_human and human and not gold_std:
            workerStats[workerId]['noGoldWrongGuessHuman']+=1
        if human==generated:
            annom+=1
        if correct:
            totalCorrect+=1
            workerStats[workerId]['correct']+=1
        else:
            approve=False
            failedTrans=True
        if gold_std:
            totalGold+=1
            workerStats[workerId]['goldTotal']+=1
            if generated:
                goldRight+=1
                workerStats[workerId]['goldRight']+=1
            else:
                approve=False
                failedGold=True
        if gt_human:
            workerStats[workerId]['eReal']+=1
        else:
            workerStats[workerId]['eGen']+=1

        
        #allRows.append((row,[workerId,accept
        if approve:
            row.append('Approve')
        elif failedGold:
            row+=['','Misclassified obvious image (diliberately poor quality)']
        elif failedTrans:
            row+=['','Failed attention task (selected incorrect transcription)']
        allRows.append(row)


with open('app_rej.csv','w') as out:
    csvwriter = csv.writer(out, delimiter=',',
            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(header)
    for row in allRows:
        csvwriter.writerow(row)
        


print('Total')
print('Correct: {}, guess right: {}, gold: {}/{}'.format(totalCorrect/total,totalGuessRight/total,goldRight,totalGold))
print('annom {}/{}'.format(annom,total))
#print(workerStats)

filteredWorkers=0
filteredRightGuessHuman=0.0
filteredWrongGuessHuman=0.0
filteredRightGuessGen=0.0
filteredWrongGuessGen=0.0

keptG=0
keptT=0
reG=0
reT=0

suspect=[]
accWrkr=[]
accWrkrId=[]

for workerId,stats in workerStats.items():
    print('{}\t{}\t{}\t{}\tcorrect:{}\tright:{}'.format(workerId,stats['total'],workerStats[workerId]['eReal'],workerStats[workerId]['eGen'],stats['correct']/stats['total'],stats['guessRight']/stats['total']))
    if stats['goldTotal']>0:
        print('\t\tgold:{}/{}'.format(stats['goldRight'],stats['goldTotal']))
        goldRatio = stats['goldRight']/stats['goldTotal']
    else:
        goldRatio = 1
    correctRatio = stats['correct']/stats['total']

    #if stats['guessRight']/stats['total']>0.5:
    #if stats['total']-stats['goldTotal']>0 and (stats['noGoldRightGuessHuman']+stats['noGoldRightGuessGen'])/(stats['total']-stats['goldTotal'])>0.5:
    #if correctRatio>0.9 and goldRatio>0.95 and stats['total']>5:
    if correctRatio>0.9 and stats['total']>5: #final crit
        filteredWorkers+=1
        filteredRightGuessHuman+=stats['noGoldRightGuessHuman']
        filteredWrongGuessHuman+=stats['noGoldWrongGuessHuman']
        filteredRightGuessGen+=stats['noGoldRightGuessGen']
        filteredWrongGuessGen+=stats['noGoldWrongGuessGen']

        accWrkr.append( (stats['noGoldRightGuessGen']+stats['noGoldRightGuessHuman'])/((stats['noGoldRightGuessGen']+stats['noGoldRightGuessHuman']+stats['noGoldWrongGuessGen']+stats['noGoldWrongGuessHuman'])))
        accWrkrId.append(workerId)

        keptG+=stats['goldRight']
        keptT+=stats['goldTotal']
    else:#if stats['total']<3:
        reG+=stats['goldRight']
        reT+=stats['goldTotal']

    if correctRatio<0.75 or (stats['goldTotal']>4 and goldRatio<0.5):
        suspect.append((workerId,stats))


filteredTotal = filteredRightGuessHuman+filteredWrongGuessHuman+filteredRightGuessGen+filteredWrongGuessGen
filteredRight = filteredRightGuessHuman+filteredRightGuessGen

print('kept instances:{}/{} {:.3}, workers: {:.3}\twere right: {:.3}'.format(filteredTotal,total,filteredTotal/total,filteredWorkers/len(workerStats),filteredRight/filteredTotal))
print('           Guessed: human  computer')
print('Actually human:     {:.3}  {:.3}'.format(filteredRightGuessHuman/filteredTotal,filteredWrongGuessGen/filteredTotal))
print('Actually generated: {:.3}  {:.3}'.format(filteredWrongGuessHuman/filteredTotal,filteredRightGuessGen/filteredTotal))

print('worker std dev: {}'.format(np.std(accWrkr)))
accWrkr=list(zip(accWrkr,accWrkrId))
accWrkr.sort(reverse=True,key=lambda a:a[0])
bestWrkr = accWrkr[:len(accWrkr)//10]
#print(bestWrkr)
bestScore,bestIds = zip(*bestWrkr)
filteredRightGuessHuman=0.0
filteredWrongGuessHuman=0.0
filteredRightGuessGen=0.0
filteredWrongGuessGen=0.0
#print(bestIds)
for workerId,stats in workerStats.items():
    if workerId in bestIds:
        filteredRightGuessHuman+=stats['noGoldRightGuessHuman']
        filteredWrongGuessHuman+=stats['noGoldWrongGuessHuman']
        filteredRightGuessGen+=stats['noGoldRightGuessGen']
        filteredWrongGuessGen+=stats['noGoldWrongGuessGen']

filteredTotal = filteredRightGuessHuman+filteredWrongGuessHuman+filteredRightGuessGen+filteredWrongGuessGen
filteredRight = filteredRightGuessHuman+filteredRightGuessGen
print('top 10% accuracy: {}'.format(filteredRight/filteredTotal))
print('Actually human:     {:.3}  {:.3}'.format(filteredRightGuessHuman/filteredTotal,filteredWrongGuessGen/filteredTotal))
print('Actually generated: {:.3}  {:.3}'.format(filteredWrongGuessHuman/filteredTotal,filteredRightGuessGen/filteredTotal))
print('Top 10% GT Gen: {}, GT human: {}'.format(filteredRightGuessHuman+filteredWrongGuessGen,filteredWrongGuessHuman+filteredRightGuessGen))


print(keptG/keptT)
print(reG/reT)


#print('suspect')
#for workerId,stats in suspect:
#    print('{}\t{}\t{}\t{}\tcorrect:{}\tright:{}'.format(workerId,stats['total'],workerStats[workerId]['eReal'],workerStats[workerId]['eGen'],stats['correct']/stats['total'],stats['guessRight']/stats['total']))
#    if stats['goldTotal']>0:
#        print('\t\tgold:{}/{}'.format(stats['goldRight'],stats['goldTotal']))
#print(header)
