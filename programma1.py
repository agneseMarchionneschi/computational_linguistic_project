import sys 
import codecs
import nltk
import math
import ssl
from nltk import bigrams
from math import log

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

#funzione per contare i caratteri
def contaChar(listaTokens):
	caratteri = 0.0
	for token in listaTokens:
		caratteri = caratteri + len(token)
	return caratteri

#funzione per contare gli hapax	
def hapaxes (token_TOT,vocabolario):
	hapax = 0
	for tok in vocabolario:
		frequenza = token_TOT.count(tok)
		if frequenza == 1:
			hapax = hapax + 1
	return hapax

def hapaxIncrementali (token_totali):
	hapaxTOT = []
 	porzioneTokens = [] # lista che contiene 1000 token
 	voc_porzione = [] # lista che contiene le parole tipo nelle porzioni di 1000 token
	porzioni = [token_totali[n:n+1000] for n in range(0, len(token_totali),1000)] #lista contentente le porzioni generate di 1000 token 
	for lista in porzioni: # scorro le porzioni di 1000 token
		porzioneTokens = porzioneTokens + lista # concateno la lista di 1000 tokens con la lista della porzioni
		voc_porzione = set(porzioneTokens) # calcolo il vocabolario ogni volta che aggiungo 1000 token alla lista incremento
		numHapax = hapaxes(porzioneTokens, voc_porzione) # richiamo la funzione per il conteggio deglio hapaxes su porzioni incrementali di 1000 tokens
		hapaxTOT.append(numHapax)# appendo il numero trovato nella posizione i-esima 
	return hapaxTOT
#divido il testo in token e assegno loro un pos con la funzione pos_tag
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

#calcolo il rapporto tra sostantivi e verbi creando due liste 
def rapportoSosVer(sequenzaPOS):
	n = 0.0 
	v = 0.0
	#lista sostantivi
	for element in sequenzaPOS:
		if element == ("NN" or "NNS" or "NP" or "NPS"):
			n = n + 1 
		#lista verbi
		if element == ("VB" or "VBD" or "VBG" or "VBN" or "VBN" or "VBP" or "VBZ"):
			v = v + 1
	rap = float(n)/float(v)
	return rap

def dieciPOS(distribuzioneFreqPos):
	piuFreq = distribuzioneFreqPos.most_common(10) #estrae i primi 10 elementi all'interno della lista degli elementi ordinati per frequenza
	for elem in piuFreq:
		print elem[0], "frequenza:", elem[1]

# Funzione che cacola la frequenza attesa di un bigramma
def frequenzaAttesa (x, y, N):
	FA = ((x*1.0)*(y*1.0))/(N*1.0)
	return FA

def probCondizionata (sequenzaPOS, bigrammiPOSLista):
	listaBigrammaPCOND = []
	LMI = 0.0
	N = len(sequenzaPOS)
	listaLMI = []
	distribuzioneDiFrequenza = nltk.FreqDist(bigrammiPOSLista) # distribuzione di frequenza degli elementi all'interno della lista
	# calcolo della probabilita' condizionata sui bigrammi di PoS
	for bigramma in distribuzioneDiFrequenza:
		# Frequenza osservata 
	 	frequenzaOsservata = distribuzioneDiFrequenza[bigramma]
	 	frequenza_a = sequenzaPOS.count(bigramma[0])
		frequenza_b = sequenzaPOS.count(bigramma[1])
		frequenza_Attesa = frequenzaAttesa(frequenza_a, frequenza_b, N)
	 	# Probabilita' condizionata
		PCOND = (frequenza_Attesa/N*0.1)*100
	 	listaBigrammaPCOND.append([PCOND, bigramma])
	 	#LMI = (frequenza_osservata)*math.log((frequenzaOsservata*1.0)/(frequenza_attesa), 2)
	 	LMI = (frequenzaOsservata*1.0)*math.log((frequenzaOsservata*1.0)/(frequenza_Attesa), 2)
		listaLMI.append([LMI, bigramma])
	listaBigrammaPCOND = ordina(listaBigrammaPCOND)
	listaLMI = ordina(listaLMI)
	# return primi 10 elementi
	return listaBigrammaPCOND[:10], listaLMI[:10]

def ordina (lista):
	return sorted(lista, reverse = True)


def main(file1,file2):
	sys.stdout = ("output1.txt","w")
	#leggo il file
 	fileInput1 = codecs.open(file1, "r", "utf-8")
 	fileInput2 = codecs.open(file2, "r", "utf-8")
 	raw1 = fileInput1.read()
 	raw2 = fileInput2.read()
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
 	#chiamo la funzione lunghezzaMediaChar
 	caratteri1 = contaChar(listaTokens1)
 	caratteri2 = contaChar(listaTokens2)
 	#grandezza del vocabolario
 	vocabolarioLista1 = set(listaTokens1)
 	vocabolarioLista2 = set(listaTokens2)
 	vocabolario1 = len(vocabolarioLista1)
 	vocabolario2 = len(vocabolarioLista2)
 	#calcolo la lunghezza media delle parole in termini di caratteri
 	mediaCaratteri1 = caratteri1/tokenTOTALI1
 	mediaCaratteri2 = caratteri2/tokenTOTALI2
 	#	chiamo la funzione per trovare gli hapax
 	hapaxes1 = hapaxes(listaTokens1,vocabolarioLista1)
 	hapaxes2 = hapaxes(listaTokens2,vocabolarioLista2)
 	#	hapax per porzioni incrementali di 1000 token
 	hapaxIncrementali1 = hapaxIncrementali(listaTokens1)
 	hapaxIncrementali2 = hapaxIncrementali(listaTokens2)
 	# 	POS
	testoAnalizzatoPOS1 = annotazioneLinguistica(frasi1)
	testoAnalizzatoPOS2 = annotazioneLinguistica(frasi2)
	#	LISTA POS 
	sequenzaPOS1 = estraiSequenzaPOS(testoAnalizzatoPOS1)
	sequenzaPOS2 = estraiSequenzaPOS(testoAnalizzatoPOS2)
	#	RAPPORTO SOST/VERB
	rapportoSosVer1 = rapportoSosVer(sequenzaPOS1)
	rapportoSosVer2 = rapportoSosVer(sequenzaPOS2)
	#calcolo la distribuzione di frequenza delle POS
	distribuzioneFreqPos1 = nltk.FreqDist(sequenzaPOS1)
	distribuzioneFreqPos2 = nltk.FreqDist(sequenzaPOS2)
	#estraggo coppie <token-PoS, token-PoS>)
	bigrammiPOSLista1 = bigrams(sequenzaPOS1) 
	bigrammiPOSLista2 = bigrams(sequenzaPOS2)
	# Probabilita' condizionata
	dieciBigrammiPCond1, lista_LMI1 = probCondizionata(sequenzaPOS1, bigrammiPOSLista1)
	dieciBigrammiPCond2, lista_LMI2 = probCondizionata(sequenzaPOS2, bigrammiPOSLista2)
	
 	print "Il libro Oliver Twist contiene ", lunghezzaToken1, " token, e", lunghezzaFrasi1, " frasi\n"
 	print "Il libro A Christmas Carol contiene ", lunghezzaToken2, " token, e", lunghezzaFrasi2, " frasi\n"
 	print "La lunghezza media delle frasi in termini di token in Oliver Twist e' ", lunghezzaMediaFrasi1 
 	print "La lunghezza media delle frasi in termini di token in A Christmas Carol e' ", lunghezzaMediaFrasi2
 	print "La lunghezza media delle parole in termini di caratteri in Oliver Twist e' ", mediaCaratteri1
 	print "La lunghezza media delle parole in termini di caratteri in A Christmas Carol e' ", mediaCaratteri2
 	print "Il vocabolario in Oliver Twist conta' ", vocabolario1, "parole tipo\n"
 	print "Il vocabolario in Oliver Twist conta' ", vocabolario2, "parole tipo\n"
 	print "Il vocabolario in A Christmas Carol conta' ", vocabolario2, "parole tipo\n"
 	# HAPAX
	print "\n\n- INCREMENTO DEGLI HAPAX OGNI 1000 TOKEN - \n"
	print "Oliver Twist", "\t\t", "A CHRISTMAS CAROL"
	for e1, e2 in zip(hapaxIncrementali1, hapaxIncrementali2):
		print " - %-20s" % (e1),"- %-20s" % (e2)
	print "Il rapporto sostantivi verbi in Oliver Twist e' pari a", rapportoSosVer1
 	print "Il rapporto sostantivi verbi in A Christmas Carol e' pari a", rapportoSosVer2
 	print "\n\n Stampo le 10 PoS (Part-of-Speech) piu' frequenti in Oliver Twist e la relativa frequenza:\n"
	dieciPOS(distribuzioneFreqPos1)
	print "\n\n Stampo le 10 PoS (Part-of-Speech) piu' frequenti in A Christmas Carol e la relativa frequenza:\n"
	dieciPOS(distribuzioneFreqPos2)
	# CONDIZIONATA
	print "\n\n- I 10 BIGRAMMI DI POS TAG CON PROBABILITA' CONDIZIONATA MASSIMA -\n"
	print "Oliver Twist", "\t\t\t\t\t\t\t\t\t", "A CHRISTMAS CAROL"
	for elemento1, elemento2 in zip(dieciBigrammiPCond1, dieciBigrammiPCond2):
		print " Bigramma --> %-3s - %-20s  P.Condizionata --> %-0s %-20s" % (elemento1[1][0],elemento1[1][1],"%1.2f" % elemento1[0], "%")," Bigramma --> %-3s - %-20s  P.Condizionata  --> %-0s %-20s" % (elemento2[1][0], elemento2[1][1], "%1.2f" % elemento2[0], "%")
	# LMI
	print "\n\n- I 10 BIGRAMMI DI POS TAG con FORZA ASSOCIATIVA MASSIMA E LA RELATIVA FORZA ASSOCIATIVA -\n"
	print "Oliver Twist", "\t\t\t\t\t\t\t\t\t", "A CHRISTMAS CAROL"
	for elemento1, elemento2 in zip(lista_LMI1, lista_LMI2):
		print " \nbigramma --> %-10s LMI --> %-20s" % (elemento1[1], elemento1[0]),"\t\t\t\tbigramma --> %-10s LMI --> %-0s" % (elemento2[1], elemento2[0])
main(sys.argv[1], sys.argv[2])
 	

 	
