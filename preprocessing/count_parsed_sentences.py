a = []
for i in range(1, 54):
  with open("parsed_sentence_pairs/ALL_parsed_sentence_pairs_{}.txt".format(i), "rb") as f:
    a += f.readlines()

a = list(set(a))

with open("parsed_sentence_pairs/ALL_parsed_sentence_pairs_{}.txt", "a") as w:
  for line in a:
    w.write(line)

print len(a)

markers = {}

for line in a:
  datum = line[:-1].split("\t")
  if len(datum) == 3:
    marker = datum[2]
    if marker in markers:
      markers[marker] += 1
    else:
      markers[marker] = 1

print markers

print "and     y        {: >8d}".format(markers["y"])
print "but     pero     {: >8d}".format(markers["pero"])
print "because porque   {: >8d}".format(markers["porque"])
print "if      si       {: >8d}".format(markers["si"])
print "when    cuando   {: >8d}".format(markers["cuando"])
print "so      entonces {: >8d}".format(markers["entonces"])
print "before  antes    {: >8d}".format(markers["antes"])
print "though  aunque   {: >8d}".format(markers["aunque"])

