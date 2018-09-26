"""
Take the flattened files and turn into DrQA format
line
{"id": "doc1", "text": "text of doc1"}
...
{"id": "docN", "text": "text of docN"}
"""
import json

check_repeat = set()

full = False


def check_because(line):
    line = line.lower()
    if 'because' in line and 'because of' not in line:
        return True
    else:
        return False


def line_to_json(f, line, i, prefix):
    f.write(json.dumps({'id': prefix + str(i), 'text': line.strip()}) + '\n')
    # f.write('{"id": "' + prefix + str(i) + '", "text":"' + line.strip().replace('"', '\\"') + '"' '} \n')


def print_range(buffer_size=10):
    with open('./corpus/because/because_db_buffer10.txt', 'w') as file:
        buffer = []
        with open('./corpus/gigaword_en/gigaword_en_flattened.txt', 'r') as gigaword:
            total = 116456445

            found_because = False
            for i, line in enumerate(gigaword):
                if line.strip() not in check_repeat:
                    check_repeat.add(line.strip())

                    # 1. check for because
                    if check_because(line):
                        # 1.1. if because is there, clear buffer, add everything
                        buffer.append(line.strip())
                        for offset, context in enumerate(buffer):
                            line_to_json(file, context, i - (len(buffer) - offset), prefix="gigaword_")
                        buffer = []
                        found_because = True
                    else:
                        # 2. if because is not there, check buffer size
                        if len(buffer) == buffer_size:
                            # 2.1. if equal to buffer size and we discovered because 10 sents before
                            #      then we still add to file
                            if found_because:
                                for offset, context in enumerate(buffer):
                                    line_to_json(file, context, i - (len(buffer) - offset), prefix="gigaword_")
                                buffer = []
                                found_because = False
                            else:
                                buffer.pop(0)
                                buffer.append(line.strip())
                        else:
                            buffer.append(line.strip())

                if i % 500000 == 0:
                    print("gigaword {} / {}, {:3f}".format(i, total, float(i) / total * 100))

        print("loaded gigaword")

        buffer = []
        with open('./corpus/news_crawl/news_crawl_0717_flattened.txt', 'r') as newsflat:
            total = 191599627
            found_because = False

            for i, line in enumerate(newsflat):
                if line.strip() not in check_repeat:
                    check_repeat.add(line.strip())

                    # 1. check for because
                    if check_because(line):
                        # 1.1. if because is there, clear buffer, add everything
                        buffer.append(line.strip())
                        for offset, context in enumerate(buffer):
                            line_to_json(file, context, i - (len(buffer) - offset), prefix="newscrawl_")
                        buffer = []
                    else:
                        # 2. if because is not there, check buffer size
                        if len(buffer) == buffer_size:
                            # 2.1. if equal to buffer size and we discovered because 10 sents before
                            #      then we still add to file
                            if found_because:
                                for offset, context in enumerate(buffer):
                                    line_to_json(file, context, i - (len(buffer) - offset), prefix="newscrawl_")
                                buffer = []
                                found_because = False
                            else:
                                buffer.pop(0)
                                buffer.append(line.strip())
                        else:
                            buffer.append(line.strip())

                if i % 500000 == 0:
                    print("news crawl {} / {}, {:3f}".format(i, total, float(i) / total * 100))
                    # print("news crawl {}".format(end_of_giga_line + i))


def print_full():
    with open('./corpus/because/because_db.txt', 'w') as file:
        with open('./corpus/gigaword_en/gigaword_en_flattened.txt', 'r') as gigaword:
            total = 116456445
            for i, line in enumerate(gigaword):
                if line.strip() not in check_repeat:
                    check_repeat.add(line.strip())
                    file.write(
                        '{"id": "gigaword_' + str(i) + '", "text":"' + line.strip().replace('"', '\\"') + '"' '} \n')

                if i % 500000 == 0:
                    print("gigaword {} / {}, {:3f}".format(i, total, float(i) / total * 100))

        print("loaded gigaword")

        with open('./corpus/news_crawl/news_crawl_0717_flattened.txt', 'r') as newsflat:
            total = 191599627
            for i, line in enumerate(newsflat):
                # if '"' not in line.strip():
                #     continue
                if line.strip() not in check_repeat:
                    check_repeat.add(line.strip())
                    file.write(
                        '{"id": "newscrawl_' + str(i) + '", "text":"' + line.strip().replace('"', '\\"') + '"' '} \n')
                if i % 500000 == 0:
                    print("news crawl {} / {}, {:3f}".format(i, total, float(i) / total * 100))


if __name__ == '__main__':
    if full:
        print_full()
    else:
        print_range()
