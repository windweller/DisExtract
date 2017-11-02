"""
(might change name later)
Based on commands, select a group of markers,
Add additional preprocessing steps for (s1, s2) here, such as delete punctuations
Filter based on length, ratio, cap (if needed), etc.
Merge them into one set, train/val/test split, np.shuffle (fix random seed)

(then Torchtext can take it from there!)
"""


def filtering(source_dir, args):

    args.min_ratio = 1/args.max_ratio

    marker_dir = pjoin(source_dir, "markers_" + DISCOURSE_MARKER_SET_TAG)
    split_dir = pjoin(marker_dir, "split_train{}".format(args.train_size))
    ssplit_dir = pjoin(split_dir, "ssplit_" + args.method)
    input_dir = pjoin(ssplit_dir, "files")

    filter_dir = pjoin(ssplit_dir, "filter_max{}_min{}_ratio{}_undersamp{}".format(
        args.max_seq_len,
        args.min_seq_len,
        args.max_ratio,
        args.undersamp_cutoff
    ))
    output_dir = pjoin(filter_dir, "files")

    if not os.path.exists(filter_dir):
        os.makedirs(filter_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    def get_data(element_type, split):
        filename = "{}_{}_{}.txt".format(args.method, split, element_type)
        file_path = pjoin(input_dir, filename)
        return open(file_path, "rU").readlines()

    frequencies = {}
    for split in ["train", "valid", "test"]:
        frequencies[split] = {}
        for marker in DISCOURSE_MARKERS:
            frequencies[split][marker] = 0

    statistics_lines = []
    for split in ["train", "valid", "test"]:
        keep = {"s1": [], "s2": [], "label": []}

        # length-based filtering
        s1s = get_data("s1", split)
        s2s = get_data("s2", split)
        labels = get_data("label", split)
        assert(len(s1s) == len(s2s) and len(s2s) == len(labels))
        for i in range(len(s1s)):
            s1 = s1s[i][:-1]
            s2 = s2s[i][:-1]
            label = labels[i][:-1]
            len1 = len(s1.split())
            len2 = len(s2.split())
            ratio = float(len2)/len1
            if args.min_seq_len<len1 and len1<args.max_seq_len \
                    and args.min_seq_len<len2 and len2<args.max_seq_len \
                    and args.min_ratio<ratio and ratio<args.max_ratio:
                keep["s1"].append(s1)
                keep["s2"].append(s2)
                keep["label"].append(label)
                frequencies[split][label] += 1

        # write new filtered files
        for element_type in ["s1", "s2", "label"]:
            filename = "{}_{}_{}_{}_{}_{}.txt".format(
                split, 
                element_type,
                args.method, 
                args.max_seq_len, 
                args.min_seq_len, 
                args.max_ratio
            )
            file_path = pjoin(output_dir, filename)
            with open(file_path, "w") as write_file:
                for element in keep[element_type]:
                    write_file.write(element + "\n")

    statistics_lines = []
    for split in frequencies:
        for marker in frequencies[split]:
            freq = frequencies[split][marker]
            statistics_lines.append("{}\t{}\t{}".format(split, marker, freq))
    statistics_report = "\n".join(statistics_lines)
    open(pjoin(filter_dir, "VERSION.txt"), "w").write(
        "commit: \n\ncommand: \n\statistics:\n" + statistics_report
    )
