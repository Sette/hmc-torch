# IJCNN 2027 paper — hard limit: 6 pages including references.
LATEXMK ?= latexmk
MAIN     = main.tex
LOG      = $(MAIN:.tex=.log)
PDF      = $(MAIN:.tex=.pdf)

.PHONY: all pdf check watch deps clean distclean

all: pdf

pdf:
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(MAIN)

check: pdf
	@if grep -q "undefined" $(LOG); then \
		echo "!! undefined references/citations:"; grep -n "undefined" $(LOG); exit 1; \
	fi
	@pages=$$(pdfinfo $(PDF) | awk '/Pages/{print $$2}'); \
	echo "pages: $$pages (limit 6)"; \
	test "$$pages" -le 6 || { echo "!! over the 6-page limit"; exit 1; }
	@echo "OK"

watch:
	$(LATEXMK) -pdf -pvc $(MAIN)

deps:
	@tlmgr --usermode install ieeetran || true
	@kpsewhich IEEEtran.cls >/dev/null || { echo "missing IEEEtran.cls"; exit 1; }

clean:
	$(LATEXMK) -c $(MAIN)

distclean:
	$(LATEXMK) -C $(MAIN)
