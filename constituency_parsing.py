import stanza
from nltk import Tree
from nltk.draw.util import CanvasFrame
from nltk.draw import TreeWidget
import os

def main():
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
    set_of_responses = []
    set_of_ids = []
    # doc = nlp('This is a test. I am going to extend this ting')
    # set_of_responses.append(doc)

    with open('mistral_top_100.csv') as mistral:
        mistral_lines = mistral.readlines()

        for i in range(1, 31):
            mistral_out = mistral_lines[i].split(";")
            # print(mistral_out)
            # print(mistral_out[-1])

            parsed_output = nlp(mistral_out[-1])
            set_of_responses.append(parsed_output)
            set_of_ids.append(mistral_out[0])

            print(f"done with step {i} out of 30")

            # os.system('convert tree.ps tree.png')
            # psimage = Image.open('tree.ps')
            # psimage.save('tree.png')

    mistral.close()

    print("done with parsing")

    with open("constituency_mistral.csv", "w+") as output:
        output.write("id;sentence1;sentence2, etc\n")
        for i in range(len(set_of_ids)):
            response = set_of_responses[i]
            id = set_of_ids[i]
            output.write(f'{id}')
            # for response in set_of_responses:

            sentencenum = 1
            for sentence in response.sentences:
                # print(sentence.constituency)
                output.write(f";{sentence.constituency}")

                cf = CanvasFrame()
                t = Tree.fromstring(str(sentence.constituency))
                tc = TreeWidget(cf.canvas(), t)
                cf.add_widget(tc, 10, 10)  # (10,10) offsets
                cf.print_to_file(f'./trees/tree{id}sentence{sentencenum}.ps')
                cf.destroy()

                sentencenum += 1

            output.write("\n")
            print(f'done with writing {i + 1} to file')



    output.close()
    print("done")

if __name__ == "__main__":
    main()

