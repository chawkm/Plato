import os
import re
import RAKE

# Parameters
data_file = "test_small.txt"
output_file = "test_small_labelled"
lab_del = '\.'
max_keywords = 2

R = RAKE.Rake(RAKE.SmartStopList())
def label_sentence(s):
    if s == '.':
        return s
    keywords = [x[0] for x in R.run(s)]
    if len(keywords) > max_keywords:
        keywords = keywords[:max_keywords]
    label = (' ').join(keywords)
    label = label[0].upper() + label[1:]
    return " " + label + ": " + s


# Concatenate the text
concatenated_text = ""
with open(data_file, 'r') as content_file:
    content = content_file.read()
    sentences = re.split('(' + lab_del + '|[^' + lab_del + ']*)', content)[1::2]
    sentences = list(map(label_sentence, sentences))
    concatenated_text = ('').join(sentences)

text_file = open(output_file, "w")
text_file.write(concatenated_text)
