import chess
import chess.engine
import random
import csv

intents = [
    'dictate_move',
    'takeback_move',
    'best_move'
]

intent_dataset_probs = [
    0.6,
    0.2,
    0.2
]

slot_names = [
    'piece',
    'src_square',
    'dst_square',
    'O'
]

standard_intent_examples = {
    'best_move' :
    [
        'what is the best move here',
        'what is the correct move here',
        'best move',
        'what was the best move',
        'what is the best move',
        'correct move',
        'what was the correct move',
        'what is the correct move',
        'what should I have done',
        'what move should I have made',
        'what should I do here',
        'what is a good move here'
        'what is the winning move here'
    ],
    'takeback_move' :
    [
        'takeback',
        'undo',
        'revert'
    ]
}

pieces = ['pawn', 'bishop', 'king', 'queen', 'rook', 'knight']
def random_piece():
    return random.choice(pieces)

def random_square():
    return chr(random.randint(0, 7) + ord('a')) + '' + str(random.randint(0, 7)+1)

#three variations
    #[piece] [src_square] to [dst_square]
    #[piece] to [dst_square]
    #[src_square] to [dst_square]
#does no checking if a move is legal
def dictate_move_examples(variation = 0):
    if(variation == 0):
        return ("{} {} to {}".format(random_piece(), random_square(), random_square()), \
                get_slots(['piece', 'src_square', 'O', 'dst_square']))
    elif (variation == 1):
        return ("{} to {}".format(random_piece(), random_square()), \
                get_slots(['piece', 'O', 'dst_square']))
    elif (variation == 2):
        return ("{} to {}".format(random_square(), random_square()), \
                    get_slots(['src_square', 'O', 'dst_square']))

def get_slots (slot_name_list):
    return " ".join([str(slot_idx[n]) for n in slot_name_list])

#somewhat inefficient
def all_tokens_to_null_slot(txt):
    return " ".join([str(slot_idx['O']) for w in txt.split()])

def get_nlu_set (size):
    intent = []
    slots = []
    random_intent_idxs = random.choices(range(0, len(intents)), weights=intent_dataset_probs, k=size)
    for i in range(0, size):
        idx = random_intent_idxs[i]
        intent_name = intents[idx]
        if intent_name == 'dictate_move':
            ex, ex_slot = dictate_move_examples(random.randint(0, 2))
        else:
            ex = random.choice(standard_intent_examples[intent_name])
            ex_slot = all_tokens_to_null_slot(ex)
        intent.append((ex, idx))
        slots.append(ex_slot)
    return (intent, slots)

def init_slot_idx():
    idx = 0
    for slot_name in slot_names:
        slot_idx[slot_name] = idx
        idx += 1

def save_intents(path):
    file_name = 'dict.intents.csv'
    f = open(path+file_name, 'w', newline='')
    writer = csv.writer(f)
    for intent in intents:
        writer.writerow([intent])
    f.close()

def save_slots(path):
    file_name = 'dict.slots.csv'
    f = open(path+file_name, 'w', newline='')
    writer = csv.writer(f)
    for slot in slot_names:
        writer.writerow([slot])
    f.close()

def save_train_data(path, file_name_prefix, data, data_slots):
    f = open(path + file_name_prefix+'.tsv', 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(['sentence \t label'])
    for entry in data:
        writer.writerow(["{}\t{}".format(entry[0], entry[1])])
    f.close()

    f = open(path + file_name_prefix+'_slots.tsv', 'w', newline='')
    writer = csv.writer(f)
    for entry in data_slots:
        writer.writerow([entry])
    f.close()

def generate_nlu_datasets():
    train_size = 5000
    test_size = 500
    nlu_path = 'datasets/nlu/'

    train, train_slots = get_nlu_set(train_size)
    test, test_slots = get_nlu_set(test_size)

    save_intents(nlu_path)
    save_slots(nlu_path)
    save_train_data(nlu_path, 'train', train, train_slots)
    save_train_data(nlu_path, 'test', test, test_slots)


slot_idx = dict()
init_slot_idx()
generate_nlu_datasets()





