"""
Take the flattened files and turn into DrQA format
line
{"id": "doc1", "text": "text of doc1"}
...
{"id": "docN", "text": "text of docN"}
"""
check_repeat = set()

with open('./corpus/because/because_db_2.txt', 'w') as file:
    with open('./corpus/gigaword_en/gigaword_en_flattened.txt', 'r') as gigaword:
        for i, line in enumerate(gigaword):
            if line.strip() not in check_repeat:
                check_repeat.add(line.strip())
                file.write('{"id": "doc_' + str(i) + '", "text":"' + line.replace('"', '\\"') + '"' '} \n')

            if i % 100000 == 0:
                print("gigaword {}".format(i))

        end_of_giga_line = i
    print("loaded gigaword")

    with open('./corpus/news_crawl/news_crawl_0717_flattened.txt', 'r') as newsflat:
        for i, line in enumerate(newsflat):
            # if '"' not in line.strip():
            #     continue
            if line.strip() not in check_repeat:
                check_repeat.add(line.strip())
                file.write('{"id": "doc_' + str(end_of_giga_line + i) + '", "text":"' + line.replace('"', '\\"') + '"' '} \n')

            if i % 100000 == 0:
                print("news crawl {}".format(end_of_giga_line + i))
