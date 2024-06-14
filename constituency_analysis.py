import pandas as pd

import stanza


def get_from_tree(tree, target_string:str) -> (bool, list, list) :
    # print(len(tree.children))
    if len(tree.children) == 0:
        if tree.label == target_string:
            return True, [], []
        else:
            return False, [], []

    elif len(tree.children) == 1:
        isgood, childres, childlabel = get_from_tree(tree.children[0], target_string)
        if isgood:
            if len(childlabel) == 0:
                return isgood, childres, childlabel + [tree.label]
            else:
                return isgood, childres, childlabel
        else:
            return False, childres, childlabel


    else:
        node_good = False
        res = []
        labels = []
        for child in tree.children:
            isgood, child_res, child_labels = get_from_tree(child, target_string)
            if isgood:
                node_good = True
            res += child_res
            labels += child_labels

        if node_good:
            res.append(tree)
            # print("breh", res, labels)
            return False, res, labels
        else:
            return False, res, labels


def main():
    set_of_trees = []
    tree_scores = []
    # doc = nlp('This is a test. I am going to extend this ting')
    # set_of_responses.append(doc)
    sorted_tokens = pd.read_csv('pos_tag_analysis.csv')
    top10 = sorted_tokens[0:30]['Unnamed: 0']
    # print(top10[0])

    with open('constituency_mistral_inputs.csv') as mistral:
        consti_lines = mistral.readlines()

        for i in range(1, 31):
            constituent_line = consti_lines[i].split(";")
            for j in range(2, len(constituent_line)):
                tree = stanza.models.constituency.tree_reader.read_trees(constituent_line[j])[0]
                set_of_trees.append(tree)
                print(constituent_line[1])
                tree_scores.append(constituent_line[1])
                # print(tree.children[0].label)
            print(f"Built trees for line {i} out of 30")
    mistral.close()

    # tree0 = set_of_trees[0]
    # uselessboolean, res, labels = get_from_tree(tree0, "article")
    # uselessboolean2, res2, labels2 = get_from_tree(tree0, "Fucking")
    # print(res+res2)
    # print(labels+labels2+['NNP'])

    token_list_in_order = []
    token_labels_in_order = []
    token_scores_in_order = []
    for token in top10:
        # token = top10[i]

        total_set_of_token_outcomes = []
        total_set_of_token_labels = []
        total_set_of_token_scores = []
        for i in range(len(set_of_trees)):
            tree = set_of_trees[i]
            score = float(tree_scores[i])
            # total_set_of_token_scores.append([])
            uselessboolean, subtree_list, tree_labels = get_from_tree(tree, token)

            if len(subtree_list) > 0:
                total_set_of_token_outcomes += subtree_list
                total_set_of_token_labels += tree_labels
                total_set_of_token_scores.append(score)

        token_list_in_order.append(total_set_of_token_outcomes)
        token_labels_in_order.append(total_set_of_token_labels)
        token_scores_in_order.append(total_set_of_token_scores)

    # for i in range(len(token_list_in_order)):
    #     print(token_list_in_order[i])
    #     print(token_labels_in_order[i])
    #     print(top10[i])
    #     print("\n")
    #
    print("done with parsing")
    #
    with open("constituency_outputs_real_quick.csv", "w+") as output:
        for i in range(len(token_list_in_order)):
            # response = set_of_responses[i]
            # id = set_of_ids[i]
            # output.write(f'{id}')
            output.write(f'{top10[i]};appears {len(token_list_in_order[i])} times;\n')
            output.write(f'{token_list_in_order[i]}\n')
            output.write(f'{token_labels_in_order[i]}\n')
            output.write(f'{token_scores_in_order[i]}\n')
            output.write(f'average: {sum(token_scores_in_order[i])/len(token_scores_in_order[i])}\n')

            output.write("\n")
            # for response in set_of_responses:

            # output.write("\n")
            print(f'done with writing {i + 1} to file')

    # output.close()
    # print("done")

if __name__ == "__main__":
    main()

