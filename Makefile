SOURCES := $(shell find src -name '*.tco' -type f)
.SECONDARY:;
.PHONY: run;
.PRECIOUS: build/tcomp.s1.asm;

build/tcomp.s2: build/tcomp.s1 $(SOURCES)
	$< src/main.tco $@

build/bootstrap: bootstrap/bootstrap.cpp Makefile
	@mkdir -p $(@D)
	$(CXX) -ggdb -o $@ $< -std=c++17

build/tcomp.s1.asm: build/bootstrap $(SOURCES)
	#gdb $< -ex 'start src/main.tco $@'
	$< src/main.tco $@
	#strace $< src/main.tco $@

build/tcomp.s1.obj: build/tcomp.s1.asm
	nasm $< -o $@ -felf64

build/tcomp.s1: build/tcomp.s1.obj
	ld -o $@ $<

run: build/tcomp.s1
	#strace $<
	$<
	#gdb $<
