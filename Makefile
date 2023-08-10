CC = cargo build
CFLAGS = -r --jobs=-1

DOCC = cargo doc
DOCFLAGS = --no-deps --jobs=-1
RUSTDOCFLAGS = "--html-in-header katex-header.html"

all:
	$(CC) $(CFLAGS)

doc:
	RUSTDOCFLAGS=$(RUSTDOCFLAGS) $(DOCC) $(DOCFLAGS)
