PUNCTUATION = u'.,:;—'


def capitalize(s):
    return s[0].capitalize() + s[1:]


def standardize_sentence_output(s, lang="en"):
    if len(s) == 0:
        return None
    else:
        s = s.strip()
        s = capitalize(s)

        if lang == "sp":
            # remove all punctuation
            for p in PUNCTUATION:
                # print s
                s = s.replace(p, "")
            return s
        else:
            # strip original final punctuation
            while s[-1] in (PUNCTUATION + '"\''):

                # keep final quotations and don't add an additional .
                if len(s) > 3 and s[-3:] in ", ', \". \". '":
                    return s
                else:
                    # otherwise, strip ending punct
                    s = s[:-1].strip()
                    if len(s) == 0:
                        return None
            # add new standard . at end of sentence
            return s + " ."

# https://stackoverflow.com/a/18669080
def get_indices(lst, element, case="sensitive"):
    result = []
    starting_index = -1
    while True:
        try:
            found_index = lst.index(element, starting_index + 1)
            starting_index = found_index
        except ValueError:
            return result
        result.append(found_index)


class Sentence(object):
    class Sentence():
        def __init__(self, json_sentence, original_sentence, lang):
            self.json = json_sentence
            self.dependencies = json_sentence["basicDependencies"]
            self.tokens = json_sentence["tokens"]
            self.original_sentence = original_sentence
            self.lang = lang

        def indices(self, word):
            if len(word.split(" ")) > 1:
                words = word.split(" ")
                indices = [i for lst in [self.indices(w) for w in words] for i in lst]
                return indices
            else:
                # print " ".join([t["word"].lower() for t in self.tokens])
                # print word.lower()
                return [i + 1 for i in get_indices([t["word"].lower() for t in self.tokens], word)]

        def token(self, index):
            return self.tokens[int(index) - 1]

        def word(self, index):
            return self.token(index)["word"]

        def is_punct(self, index):
            pos = self.token(index)["pos"]
            # return pos in '.,"-RRB--LRB-:;'
            return pos in PUNCTUATION

        def is_verb(self, index):
            pos = self.token(index)["pos"]
            if pos[0] == "V":
                return True
            else:
                cop_relations = self.find_deps(index, dir="children", filter_types="cop")
                has_cop_relation = len(cop_relations) > 0
                if has_cop_relation:
                    return True
                else:
                    return False

        def find_deps(self, index, dir=None, filter_types=False, exclude_types=False, exclude_type_and_POS=False):
            deps = []
            if dir == "parents" or dir == None:
                deps += [{"dep": d, "index": d['governor']} for d in self.dependencies if d['dependent'] == index]
            if dir == "children" or dir == None:
                deps += [{"dep": d, "index": d['dependent']} for d in self.dependencies if d['governor'] == index]

            if filter_types:
                deps = [d for d in deps if d["dep"]["dep"] in filter_types]
            if exclude_types:
                deps = [d for d in deps if not d["dep"]["dep"] in exclude_types]

            if exclude_type_and_POS:
                deps = [d for d in deps if (d["dep"]["dep"], self.token(d["index"])["pos"]) not in exclude_type_and_POS]

            return [d["dep"] for d in deps]

        def find_dep_types(self, index, dir=None, filter_types=False):
            deps = self.find_deps(index, dir=dir, filter_types=filter_types)
            return [d["dep"] for d in deps]