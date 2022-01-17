import sys 
import codecs
import nltk
import math
import ssl
from nltk import bigrams
from math import log
import re 
from operator import itemgetter

#funzione che mi restituisce i token totali e la lista dei token
def lunghezzaToken(frasi):
	listaTokens = []
	tokensTOT = 0.0
	for frase in frasi:
		#divido la frase in token
		tokens = nltk.word_tokenize(frase)
		#creo la lista dei token 
		listaTokens = listaTokens + tokens
		#calcolo la lunghezza
	tokensTOT = len(listaTokens)	
	return tokensTOT, listaTokens

# funzione con cui divido il testo in token e assegno loro un pos
def annotazioneLinguistica(frasi):
	tokensPOSTot = []
	tokensTOT = []
	for frase in frasi:
		tokens = nltk.word_tokenize(frase)
		tokensPOS = nltk.pos_tag(tokens)
		tokensPOSTot = tokensPOSTot + tokensPOS
	return tokensPOSTot

#inserisce elementi in coda ad una lista esistente
def estraiSequenzaPOS(testoAnalizzatoPOS):
	listaPOS = []
	for bigramma in testoAnalizzatoPOS:
		#lista POS
		listaPOS.append(bigramma[1])
	return listaPOS	

def analisiLinguistica(testoAnalizzatoPOS):
	NETlist = []
	analisi = nltk.ne_chunk(testoAnalizzatoPOS)#net 
	for nodo in analisi: #ciclo albero scorrendo nodi
		NE = ''
		if hasattr(nodo, 'label'): #controllo se il nodo e' intermedio
			if (nodo.label() in ['PERSON']): #controllo che sia un nome proprio di persona  
				for partNE  in nodo.leaves(): #ciclo foglie nel modo selezionato. partNE sono coppie (nomeProprio, POS)
					NE = NE + ' ' + partNE[0] # creo i vari elementi della NETlist finale
					NETlist.append(NE)
	return NETlist #risultato del net, nodi analizzati


def dieciNomi(netList):
	soloNomi = []
	#trovo 10 nomi propri piu' frequenti file 1
	frequenzeNomi = nltk.FreqDist(netList)
	piuFreq = frequenzeNomi.most_common(10)
	for elem in piuFreq:
		soloNomi.append(elem[0])
	return piuFreq, soloNomi

def trovaFrasi(soloNomi, frasi):
	listaFrasi = []
	for nome in soloNomi: #scorro i 10 nomi propri
		listaFrasi = []
		for sentence in frasi: #scorro le frasi del testo
			if nome in sentence: #se trovo il nome che mi interessa nella frase
				listaFrasi.append(sentence)#aggiungo frase a lista di frasi che contengono quel nome
		fraseMin = ''
		fraseMax = ''
		minimo = float('inf')# imposto il minimo
		maximo = 0.0 # imposto il massimo
		for fraseInLista in listaFrasi: #per ogni nome, scorro le frasi che lo contengono
			if len(fraseInLista) > maximo:
				maximo = len(fraseInLista) #attribuisco a max1 la lunghezza massima della frase trovata per ora
				fraseMax = fraseInLista#salvo frase
			elif len(fraseInLista) < minimo:
				minimo = len(fraseInLista) #attribuisco a max1 la lunghezza minima della frase trovata per ora
				fraseMin = fraseInLista #salvo frase
		print "\n La frase piu' lunga con nome", nome, "e':", fraseMax.encode("utf8"), "\n"
		print "\n La frase piu' corta con nome", nome, "e':", fraseMin.encode("utf8"), "\n"
	return listaFrasi



def analisiInformazioni(frasi, soloNomi):
	#creo lista tag per sostantivi
	sost = ["NN", "NNS", "NNP", "NNPS"]
	#creo lista tag per verbi
	vb = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
	listaFrasi = []
	for nome in soloNomi: #scorro i 10 nomi propri
		listaFrasi = []
		#trovo lista dei luoghi, delle persone, dei sostantivi, dei verbi testo 
		netLista = [] #creo lista per analisi net delle frasi in cui ci sono i 10 nomi propri di persona 
		sostantivi = []#lista per tutti i sostantivi delle frasi dei 10 nomi
		net_lista = []#lista come net_list1 
		verbi = []#lista per tutti i verbi delle frasi dei 10 nomi
		for sentence in frasi: #scorro frasi del testo intero
			if nome in sentence:#se trovo uno dei 10 nomi
				listaFrasi.append(sentence)#lo inserisco nella lista lista_frasi1
		toksTOT = []
		for frase in listaFrasi: #scorro tutte le frasi del testo
			toks = nltk.word_tokenize(frase)
			toksTOT = toksTOT + toks #tokenizzo il testo intero
		toksPOS = nltk.pos_tag(toksTOT) #pos taggo il testo
		analisi_net = nltk.ne_chunk(toksPOS) #eseguo analisi 
		for nodo1 in analisi_net: 
			ne1 = ''
			neA = ''
			if hasattr(nodo1, 'label'): #se il nodo ha un'etichetta
				if nodo1.label() in ['GPE']:# di luogo
					for part_ne1 in nodo1.leaves():
						ne1 = ne1 + ' ' + part_ne1[0]
						netLista.append(ne1)# aggiungo alla lista la parola
				elif nodo1.label() in ['PERSON']: #di persona
					for part_neA in nodo1.leaves():
						neA = neA + ' ' + part_neA[0]
						net_lista.append(neA) #aggiungo alla lista la parola
			elif nodo1[1] in sost:
				sostantivi.append(nodo1[0])
			elif nodo1[1] in vb:
				verbi.append(nodo1[0])
		frequenzaLuoghi = nltk.FreqDist(netLista)
		luoghiPiuFreq = frequenzaLuoghi.most_common(10) #i 10 nomi di luoghi piu' frequenti nelle liste di parole dei 10 nomi propri di persona piu' frequenti
		frequenzaNomiP = nltk.FreqDist(net_lista)
		nomiPPiuFreq = frequenzaNomiP.most_common(10) #i 10 nomi di persona piu' frequenti nelle liste di parole dei 10 nomi propri di persona
		frequenzaSostantivi = nltk.FreqDist(sostantivi)
		sostantiviPiuFreq = frequenzaSostantivi.most_common(10) #i 10 sostantivi piu' frequenti nelle liste di parole dei 10 nompi propri di persona
		frequenzaVerbi = nltk.FreqDist(verbi)
		verbiPiuFreq = frequenzaVerbi.most_common(10) #i 10 verbi piu' frequenti nelle liste di parole dei 10 nompi propri di persona
		print "\n La lista dei 10 luoghi piu' frequenti associati al nome", nome, "e':\n"
		for elep in luoghiPiuFreq:
			print elep[0], "---", "frequenza:", elep[1], "\n\n"
		print "\n La lista delle 10 persone piu' frequenti associate al nome", nome, "e':\n"
		for elep1 in nomiPPiuFreq:
			print elep1[0], "---", "frequenza:", elep1[1], "\n\n" 
		print "\n La lista dei 10 sostantivi piu' frequenti associati al nome", nome, "e':\n"
		for elep2 in sostantiviPiuFreq:
			print elep2[0], "---", "frequenza:", elep2[1], "\n\n" 
		print "\n La lista dei 10 verbi piu' frequenti associati al nome", nome, "e':\n"
		for elep3 in verbiPiuFreq:
			print elep3[0], "---", "frequenza:", elep3[1], "\n\n" 


def catenaMarkov0(listaTokens, lista_frasi): 
	lista_FraseProbab = {} # dizionario che conterra' la frase e la sua probabilita'
      	lungh_Corpus = len(listaTokens)
        freqDistTesto = nltk.FreqDist(listaTokens)# distribuzione di frequenza dei singoli token
        for frase in lista_frasi:# per ogni frase
      		probFrase = 1.0 #inizializzo la prob. a 1
       	        tokens = nltk.word_tokenize(frase) #divido ogni frase in una lista di tokens
        	for tok in tokens: #scorro ogni singolo token dentro la frase
                	if (len(tokens)>=8) and (len(tokens)<=12):  #controllo se la lunghezza della frase e' compresa tra 8 e 12
                		probabToken = freqDistTesto[tok]*1.0/lungh_Corpus*1.0 # se lo e' calcolo la probabilita' del token
              			probFrase = probFrase * probabToken #poi moltiplico la probabilita' del token attuale con la precedente per la probabilita' della frase
                        	lista_FraseProbab[frase] = probFrase #inserisco frase e probabilita' dentro un dizionario
	
   	return lista_FraseProbab


def main(file1,file2):
	
	fileInput1 = codecs.open(file1, "r", "utf-8")
	fileInput2 = codecs.open(file2, "r", "utf-8")
	raw1 = fileInput1.read()
	raw2 = fileInput2.read()					
	sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	
	#divido in frasi
	frasi1 = sent_tokenizer.tokenize(raw1)
	frasi2 = sent_tokenizer.tokenize(raw2)
	sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
 	
 	#divido i due file in frasi
 	frasi1 = sent_tokenizer.tokenize(raw1)
 	frasi2 = sent_tokenizer.tokenize(raw2)
 	
 	#calcolo il numero di frasi 
 	lunghezzaFrasi1 = len(frasi1)
 	lunghezzaFrasi2 = len(frasi2)
 	
 	#chiamo la funzione "lunghezzaToken" sul testo diviso in frasi
 	lunghezzaToken1, listaTokens1 = lunghezzaToken(frasi1)
 	lunghezzaToken2,listaTokens2 = lunghezzaToken(frasi2)
 	
 	tokenTOTALI1 = len(listaTokens1)
 	tokenTOTALI2 = len(listaTokens2)
 	 
 	#calcolo la lunghezza media delle frasi in termini di token
 	lunghezzaMediaFrasi1 = lunghezzaToken1/lunghezzaFrasi1 
 	lunghezzaMediaFrasi2 = lunghezzaToken2/lunghezzaFrasi2 

 	#	POS
	testoAnalizzatoPOS1 = annotazioneLinguistica(frasi1)
	testoAnalizzatoPOS2 = annotazioneLinguistica(frasi2)
	
	# LISTA POS 
	sequenzaPOS1 = estraiSequenzaPOS(testoAnalizzatoPOS1)
	sequenzaPOS2 = estraiSequenzaPOS(testoAnalizzatoPOS2)
	
	#Analisi Linguistica
	netList1 = analisiLinguistica(testoAnalizzatoPOS1)
	netList2 = analisiLinguistica(testoAnalizzatoPOS2)

	#10 nomi piu' frequenti
	dieciNomi1, soloNomi1 = dieciNomi(netList1)
	dieciNomi2, soloNomi2 = dieciNomi(netList2)
	
	# stampo i 10 nomi propri piu' frequenti file 1
	print "\n I dieci nomi di persona piu' freqenti in Oliver Twist sono:\n"
	for elem in dieciNomi1: 
		print elem[0], "---", " con frequenza:", elem[1]
	# stampo i 10 nomi propri piu' frequenti file 2
	print "\n I dieci nomi di persona piu' freqenti in A Christmas Carol sono:\n"
	for elem6 in dieciNomi2: 
		print elem6[0], "---", " con frequenza:", elem6[1]
	# FRASE PIU ' LUNGA E CORTA PER OGNI NOME 
	print "Per ogni nome piu' frequente in Oliver Twist, la frase piu' lunga e quella piu' corta alla quale e' associato il nome:\n"
	trovaFrasi(soloNomi1,frasi1)
	print "Per ogni nome piu' frequente in A Christmas Carol, la frase piu' lunga e quella piu' corta alla quale e' associato il nome:\n"
	trovaFrasi(soloNomi2,frasi2)

	#LISTA DELLE FRASI CHE CONTENGONO I NOMI PROPRI PIU' FREQUENTI
	listaFrasi1 = trovaFrasi (soloNomi1, frasi1)
	listaFrasi2 = trovaFrasi (soloNomi2, frasi2)	
	
	#ANALISI PER OGNI NOME DEI 10 VERBI LUOGHI SOSTANTIVI PIU' FREQUENTI
	print "\n\n Oliver Twist \n\n"
	analisiInformazioni(frasi1, soloNomi1)
	print "\n\n A Christmas Carol \n\n"
	analisiInformazioni(frasi2, soloNomi2)
	
	#Calcolo la frase con probabilita' massima con modello Markov 0 per ogni nome
	
	print "\n\n Oliver Twist\n"
	for nome1 in soloNomi1:
		lista_frasi11 = []
		for sentenceS in frasi1:
			if nome1 in sentenceS:
				lista_frasi11.append(sentenceS)
		markov_1 = catenaMarkov0(listaTokens1, lista_frasi11)
		ord_markov1 = list(sorted(markov_1.items(), key = itemgetter(1)))
		index1 = len(markov_1) - 1
		frase1_max = ord_markov1[index1]
		print "\nLa frase con il nome", nome1, "con probabilita' massima calcolata con un modello di Markov 0 e':", frase1_max, "\n"
		#cerco le date tramite le espressioni regolari del testo 1
		listaFrasi3 = []
		listaDATE = []
		listaMESI = []
		listaGIORNI = []
		for sentence3 in frasi1: 
			if nome1 in sentence3:
				listaFrasi3.append(sentence3) 
		for frase3 in listaFrasi3:
			listaDATE = re.findall(r'\b(d\d?)(\s|/|\.|-)(\d\d?|[J|j]anuary|[F|f]ebruary|[M|m]arch|[A|a]pril|[M|m]ay|[J|j]une|[J|j]uly|[A|a]ugust|[S|s]eptember|[O|o]ctober|[N|n]ovember|[D|d]ecember)(\s|//-|\.|-)(\d\d\d?\d?)?\b', frase3) #cerco in ogni frase date
			listaMESI = re.findall(r'\b([J|j]anuary|[F|f]ebruary|[M|m]arch|[A|a]pril|[M|m]ay|[J|j]une|[J|j]uly|[A|a]ugust|[S|s]eptember|[O|o]ctober|[N|n]ovember|[D|d]ecember)\b', frase3)#cerco in ogni frase mesi
			listaGIORNI = re.findall(r'\b([M|m]onday|[T|t]uesday|[W|w]ednesday|[T|t]hursday|[F|f]riday|[S|s]aturday|[S|s]unday)\b', frase3)#cerco in ogni frase giorni
		print "\n Le date trovate nelle le frasi associate a", nome1, "sono:", listaDATE
		print "\n I mesi trovati nelle le frasi associate a", nome1, "sono:", listaMESI
		print "\n I giorni trovati nelle frasi associate a", nome1, "sono:", listaGIORNI

	#Calcolo la frase con probabilita' massima con modello Markov 0 per ogni nome
	
	print "\n\n A Christmas Carol\n"
	for nome2 in soloNomi2:
		lista_frasi2 = []
		for sentence9 in frasi2:
			if nome2 in sentence9:
				lista_frasi2.append(sentence9)
		markov_2 = catenaMarkov0(listaTokens2, lista_frasi2)
		ord_markov2 = list(sorted(markov_2.items(), key = itemgetter(1)))
		index2 = len(markov_2) - 1
		frase2_max = ord_markov2[index2]
		print "\nLa frase con il nome", nome2, "con probabilita' massima calcolata con un modello di Markov 0 e':", frase2_max, "\n"
		#cerco le date tramite le espressioni regolari del testo 1
		listaFrasi4 = []
		listaDATE2 = []
		listaMESI2 = []
		listaGIORNI2 = []
		for sentence8 in frasi2: 
			if nome2 in sentence8:
				listaFrasi4.append(sentence8)
		for frase8 in listaFrasi4:
			listaDATE2 = re.findall(r'\b(d\d?)(\s|/|\.|-)(\d\d?|[J|j]anuary|[F|f]ebruary|[M|m]arch|[A|a]pril|[M|m]ay|[J|j]une|[J|j]uly|[A|a]ugust|[S|s]eptember|[O|o]ctober|[N|n]ovember|[D|d]ecember)(\s|//-|\.|-)(\d\d\d?\d?)?\b', frase8) 
			listaMESI2 = re.findall(r'\b([J|j]anuary|[F|f]ebruary|[M|m]arch|[A|a]pril|[M|m]ay|[J|j]une|[J|j]uly|[A|a]ugust|[S|s]eptember|[O|o]ctober|[N|n]ovember|[D|d]ecember)\b', frase8)
			listaGIORNI2 = re.findall(r'\b([M|m]onday|[T|t]uesday|[W|w]ednesday|[T|t]hursday|[F|f]riday|[S|s]aturday|[S|s]unday)\b', frase8)
		print "\n Le date trovate nelle le frasi associate a", nome2, "sono:", listaDATE2
		print "\n I mesi trovati nelle le frasi associate a", nome2, "sono:", listaMESI2
		print "\n I giorni trovati nelle frasi associate a", nome2, "sono:", listaGIORNI2
main(sys.argv[1], sys.argv[2])







