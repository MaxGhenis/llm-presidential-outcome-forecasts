.PHONY: all clean

all: paper.pdf

paper.pdf: main.tex sections/*.tex references.bib
	pdflatex main
	bibtex main
	pdflatex main
	pdflatex main

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.pdf