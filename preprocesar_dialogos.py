def parse_dialogs_per_response(lines):
    data = []
    facts_temp = []
    utterance_temp = None
    response_temp = None
    # Parse line by line
    for line in lines:
        line = line.strip()
        if line:
            nid, line = line.split(' ', 1)
            if '\t' in line:  # Has utterance and respone
                utterance_temp, response_temp = line.split('\t')

                data.append(str(response_temp).strip())
    return data


def llenar_archivos(lines, lista_archivos):
    for archivo in lista_archivos:
        with open(archivo, 'w', encoding='utf-8') as f:
            for line in lines:
                line = line.strip()
                if line:
                    nid, line = line.split(' ', 1)
                    if '@' in line:  # Has utterance and respone
                        line = str(line).replace('@', '\t')

                    f.write(nid + ' ' + line + '\n')


if __name__ == '__main__':

    archivos = ['dialog-babi/dialog-trn.txt', 'dialog-babi/dialog-test.txt', 'dialog-babi/dialog-dev.txt']
    with open('dialog-babi/dialogos_original.txt', encoding='utf-8') as f:
        llenar_archivos(f.readlines(), archivos)

    with open('dialog-babi/dialog-trn.txt', encoding='utf-8') as f:
        cand_trn = parse_dialogs_per_response(f.readlines())

    test_data = []
    with open('dialog-babi/dialog-test.txt', encoding='utf-8') as f:
        cand_text = parse_dialogs_per_response(f.readlines())

    val_data = []
    with open('dialog-babi/dialog-dev.txt', encoding='utf-8') as f:
        cand_dev = parse_dialogs_per_response(f.readlines())

    candidatos = cand_dev + cand_text + cand_trn
    items = set(candidatos)
    with open('dialog-babi/dialog-candidates.txt', 'w', encoding='utf-8') as f:
        for i, candidate in enumerate(items):
            f.write('1 {}'.format(candidate) + '\n')
