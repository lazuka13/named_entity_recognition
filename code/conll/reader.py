from nltk.tree import Tree
from nltk.util import LazyMap, LazyConcatenation
from nltk.tag import map_tag

from nltk.corpus.reader.api import *


class DataReader(CorpusReader):
    """
    Отвечает за считывание датасета в формате CoNLL2003
    """

    WORDS = 'words'
    POS = 'pos'
    TREE = 'tree'
    CHUNK = 'chunk'
    NE = 'ne'
    SRL = 'srl'
    IGNORE = 'ignore'
    OFFSET = 'offset'
    LEN = 'len'

    COLUMN_TYPES = (WORDS, POS, TREE, CHUNK, NE, SRL, IGNORE, OFFSET, LEN)

    def __init__(self, root, fileids, columntypes,
                 chunk_types=None, root_label='S', pos_in_tree=False,
                 srl_includes_roleset=True, encoding='utf8',
                 tree_class=Tree, tagset=None):
        for columntype in columntypes:
            if columntype not in self.COLUMN_TYPES:
                raise ValueError('Bad column type %r' % columntype)
        if isinstance(chunk_types, string_types):
            chunk_types = [chunk_types]
        self._chunk_types = chunk_types
        self._colmap = dict((c, i) for (i, c) in enumerate(columntypes))
        self._pos_in_tree = pos_in_tree
        self._root_label = root_label  # for chunks
        self._srl_includes_roleset = srl_includes_roleset
        self._tree_class = tree_class
        CorpusReader.__init__(self, root, fileids, encoding)
        self._tagset = tagset

    def raw(self, fileids=None):
        if fileids is None:
            fileids = self._fileids
        elif isinstance(fileids, string_types):
            fileids = [fileids]
        return concat([self.open(f).read() for f in fileids])

    def words(self, fileids=None):
        self._require(self.WORDS)
        return LazyConcatenation(LazyMap(self._get_words, self._grids(fileids)))

    def sents(self, fileids=None):
        self._require(self.WORDS)
        return LazyMap(self._get_words, self._grids(fileids))

    def docs(self, fileids=None):
        self._require(self.WORDS)
        return LazyMap(self._get_words_sents, self._grids_sents(fileids))

    def tagged_words(self, fileids=None, tagset=None):
        self._require(self.WORDS, self.POS)

        def get_tagged_words(grid):
            return self._get_tagged_words(grid, tagset)

        return LazyConcatenation(LazyMap(get_tagged_words,
                                         self._grids(fileids)))

    def tagged_sents(self, fileids=None, tagset=None):
        self._require(self.WORDS, self.POS)

        def get_tagged_words(grid):
            return self._get_tagged_words(grid, tagset)

        return LazyMap(get_tagged_words, self._grids(fileids))

    def chunked_words(self, fileids=None, chunk_types=None,
                      tagset=None):
        self._require(self.WORDS, self.POS, self.CHUNK)
        if chunk_types is None: chunk_types = self._chunk_types

        def get_chunked_words(grid):  # capture chunk_types as local var
            return self._get_chunked_words(grid, chunk_types, tagset)

        return LazyConcatenation(LazyMap(get_chunked_words,
                                         self._grids(fileids)))

    def get_tags(self, fileids=None, tagset=None, tags=[]):
        required = []
        for tag in tags:
            if tag == 'offset':
                required.append(self.OFFSET)
            if tag == 'len':
                required.append(self.LEN)
            if tag == 'words':
                required.append(self.WORDS)
            if tag == 'pos':
                required.append(self.POS)
            if tag == 'tree':
                required.append(self.TREE)
            if tag == 'ne':
                required.append(self.NE)
            if tag == 'srl':
                required.append(self.SRL)
            if tag == 'ignore':
                required.append(self.IGNORE)
            if tag == 'chunk':
                required.append(self.CHUNK)

        self._require(*required)

        def get_tags_inn(grid, tags=tags):
            return self._get_tags(grid, tagset, tags=tags)

        return LazyConcatenation(LazyMap(get_tags_inn, self._grids(fileids)))

    def _get_tags(self, grid, tagset=None, tags=None):
        columns = [self._get_column(grid, self._colmap[tag]) for tag in tags]
        return list(zip(*columns))

    def chunked_sents(self, fileids=None, chunk_types=None,
                      tagset=None):
        self._require(self.WORDS, self.POS, self.CHUNK)
        if chunk_types is None: chunk_types = self._chunk_types

        def get_chunked_words(grid):  # capture chunk_types as local var
            return self._get_chunked_words(grid, chunk_types, tagset)

        return LazyMap(get_chunked_words, self._grids(fileids))

    def parsed_sents(self, fileids=None, pos_in_tree=None, tagset=None):
        self._require(self.WORDS, self.POS, self.TREE)
        if pos_in_tree is None: pos_in_tree = self._pos_in_tree

        def get_parsed_sent(grid):  # capture pos_in_tree as local var
            return self._get_parsed_sent(grid, pos_in_tree, tagset)

        return LazyMap(get_parsed_sent, self._grids(fileids))

    def srl_spans(self, fileids=None):
        self._require(self.SRL)
        return LazyMap(self._get_srl_spans, self._grids(fileids))

    def srl_instances(self, fileids=None, pos_in_tree=None, flatten=True):
        self._require(self.WORDS, self.POS, self.TREE, self.SRL)
        if pos_in_tree is None: pos_in_tree = self._pos_in_tree

        def get_srl_instances(grid):  # capture pos_in_tree as local var
            return self._get_srl_instances(grid, pos_in_tree)

        result = LazyMap(get_srl_instances, self._grids(fileids))
        if flatten: result = LazyConcatenation(result)
        return result

    def iob_words(self, fileids=None, tagset=None):
        """
        :return: a list of word/tag/IOB tuples
        :rtype: list(tuple)
        :param fileids: the list of fileids that make up this corpus
        :type fileids: None or str or list
        """
        self._require(self.WORDS, self.POS, self.CHUNK)

        def get_iob_words(grid):
            return self._get_iob_words(grid, tagset)

        return LazyConcatenation(LazyMap(get_iob_words, self._grids(fileids)))

    def iob_sents(self, fileids=None, tagset=None):
        """
        :return: a list of lists of word/tag/IOB tuples
        :rtype: list(list)
        :param fileids: the list of fileids that make up this corpus
        :type fileids: None or str or list
        """
        self._require(self.WORDS, self.POS, self.CHUNK)

        def get_iob_words(grid):
            return self._get_iob_words(grid, tagset)

        return LazyMap(get_iob_words, self._grids(fileids))

    def _grids(self, fileids=None):
        return concat([StreamBackedCorpusView(fileid, self._read_grid_block,
                                              encoding=enc)
                       for (fileid, enc) in self.abspaths(fileids, True)])

    def _read_grid_block(self, stream):
        grids = []
        for block in read_blankline_block(stream):
            block = block.strip()
            if not block: continue

            grid = [line.split() for line in block.split('\n')]

            # If there's a docstart row, then discard. ([xx] eventually it
            # would be good to actually use it)
            if grid[0][self._colmap.get('words', 0)] == '-DOCSTART-':
                del grid[0]

            # Check that the grid is consistent.
            for row in grid:
                if len(row) != len(grid[0]):
                    raise ValueError('Inconsistent number of columns:\n%s'
                                     % block)
            grids.append(grid)
        return grids

    def _grids_sents(self, fileids=None):
        return concat([StreamBackedCorpusView(fileid, self._read_grid_block_sents,
                                              encoding=enc)
                       for (fileid, enc) in self.abspaths(fileids, True)])

    def _read_grid_block_sents(self, stream):
        # grid here contains doc, not sent, and doc contains sent
        grids = []
        start_re = re.compile("(-DOCSTART- -X- O O)")
        for block in read_regexp_block(stream, start_re, None):
            block = block.lstrip('-DOCSTART- -X- O O\n')
            block = block.strip()
            if not block: continue
            sents = []
            for sent in block.split('\n\n'):
                sents.append([line.split() for line in sent.split('\n')])
            grids.append(sents)
        return grids

    def _get_words(self, grid):
        return self._get_column(grid, self._colmap['words'])

    def _get_words_sents(self, grid):
        return self._get_column_sents(grid, self._colmap['words'])

    def _get_tagged_words(self, grid, tagset=None):
        pos_tags = self._get_column(grid, self._colmap['pos'])
        if tagset and tagset != self._tagset:
            pos_tags = [map_tag(self._tagset, tagset, t) for t in pos_tags]
        return list(
            zip(self._get_column(grid, self._colmap['words']), pos_tags))

    def _get_iob_words(self, grid, tagset=None):
        pos_tags = self._get_column(grid, self._colmap['pos'])
        if tagset and tagset != self._tagset:
            pos_tags = [map_tag(self._tagset, tagset, t) for t in pos_tags]
        return list(zip(self._get_column(grid, self._colmap['words']), pos_tags,
                        self._get_column(grid, self._colmap['chunk'])))

    def _get_chunked_words(self, grid, chunk_types, tagset=None):
        # n.b.: this method is very similar to conllstr2tree.
        words = self._get_column(grid, self._colmap['words'])
        pos_tags = self._get_column(grid, self._colmap['pos'])
        if tagset and tagset != self._tagset:
            pos_tags = [map_tag(self._tagset, tagset, t) for t in pos_tags]
        chunk_tags = self._get_column(grid, self._colmap['chunk'])

        stack = [Tree(self._root_label, [])]

        for (word, pos_tag, chunk_tag) in zip(words, pos_tags, chunk_tags):
            if chunk_tag == 'O':
                state, chunk_type = 'O', ''
            else:
                (state, chunk_type) = chunk_tag.split('-')
            # If it's a chunk we don't care about, treat it as O.
            if chunk_types is not None and chunk_type not in chunk_types:
                state = 'O'
            # Treat a mismatching I like a B.
            if state == 'I' and chunk_type != stack[-1].label():
                state = 'B'
            # For B or I: close any open chunks
            if state in 'BO' and len(stack) == 2:
                stack.pop()
            # For B: start a new chunk.
            if state == 'B':
                new_chunk = Tree(chunk_type, [])
                stack[-1].append(new_chunk)
                stack.append(new_chunk)
            # Add the word token.
            stack[-1].append((word, pos_tag))

        return stack[0]

    def _get_parsed_sent(self, grid, pos_in_tree, tagset=None):
        words = self._get_column(grid, self._colmap['words'])
        pos_tags = self._get_column(grid, self._colmap['pos'])
        if tagset and tagset != self._tagset:
            pos_tags = [map_tag(self._tagset, tagset, t) for t in pos_tags]
        parse_tags = self._get_column(grid, self._colmap['tree'])

        treestr = ''
        for (word, pos_tag, parse_tag) in zip(words, pos_tags, parse_tags):
            if word == '(': word = '-LRB-'
            if word == ')': word = '-RRB-'
            if pos_tag == '(': pos_tag = '-LRB-'
            if pos_tag == ')': pos_tag = '-RRB-'
            (left, right) = parse_tag.split('*')
            right = right.count(')') * ')'  # only keep ')'.
            treestr += '%s (%s %s) %s' % (left, pos_tag, word, right)
        try:
            tree = self._tree_class.fromstring(treestr)
        except (ValueError, IndexError):
            tree = self._tree_class.fromstring('(%s %s)' %
                                               (self._root_label, treestr))

        if not pos_in_tree:
            for subtree in tree.subtrees():
                for i, child in enumerate(subtree):
                    if (isinstance(child, Tree) and len(child) == 1 and
                            isinstance(child[0], string_types)):
                        subtree[i] = (child[0], child.label())

        return tree

    def _get_srl_spans(self, grid):
        """
        list of list of (start, end), tag) tuples
        """
        if self._srl_includes_roleset:
            predicates = self._get_column(grid, self._colmap['srl'] + 1)
            start_col = self._colmap['srl'] + 2
        else:
            predicates = self._get_column(grid, self._colmap['srl'])
            start_col = self._colmap['srl'] + 1

        # Count how many predicates there are.  This tells us how many
        # columns to expect for SRL data.
        num_preds = len([p for p in predicates if p != '-'])

        spanlists = []
        for i in range(num_preds):
            col = self._get_column(grid, start_col + i)
            spanlist = []
            stack = []
            for wordnum, srl_tag in enumerate(col):
                (left, right) = srl_tag.split('*')
                for tag in left.split('('):
                    if tag:
                        stack.append((tag, wordnum))
                for i in range(right.count(')')):
                    (tag, start) = stack.pop()
                    spanlist.append(((start, wordnum + 1), tag))
            spanlists.append(spanlist)

        return spanlists

    def get_ne(self, fileids=None, tagset=None):
        self._require(self.NE)

        def get_ne_inn(grid):
            return self._get_ne(grid, tagset)

        return LazyConcatenation(LazyMap(get_ne_inn, self._grids(fileids)))

    def _get_ne(self, grid, tagset=None):
        return list(zip(self._get_column(grid, self._colmap['words']),
                        self._get_column(grid, self._colmap['ne'])))

    def _require(self, *columntypes):
        for columntype in columntypes:
            if columntype not in self._colmap:
                raise ValueError('This corpus does not contain a %s '
                                 'column.' % columntype)

    @staticmethod
    def _get_column(grid, column_index):
        return [grid[i][column_index] for i in range(len(grid))]

    @staticmethod
    def _get_column_sents(grid, column_index):
        return [[sent[i][column_index] for i in range(len(sent))] for sent in grid]
