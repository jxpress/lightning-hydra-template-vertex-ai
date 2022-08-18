rule dats:
     input:
         'isles.dat',
         'abyss.dat'

# delete everything so we can re-run things
rule clean:
    shell:  "rm -r logs/**"

# Count words in one of the books
rule count_words:
    input: 	'books/isles.txt'
    output: 'isles.dat'
    shell: 	'python wordcount.py books/isles.txt isles.dat'

rule count_words_abyss:
    input: 	'books/abyss.txt'
    output: 'abyss.dat'
    shell: 	'python wordcount.py books/abyss.txt abyss.dat'

rule test_conda:
    shell:
        """
        bash -c '
            . $HOME/.bashrc # if not loaded automatically
            conda activate base
            conda deactivate'
        """
