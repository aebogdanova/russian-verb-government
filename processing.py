import os
import pathlib
import re
import math
import logging
import configparser
import json
import conllu
import minio
import pickle
from tqdm import tqdm
from collections import Counter
import pandas as pd
import threading  
from concurrent.futures.thread import ThreadPoolExecutor
import pymorphy2
from pymorphy2 import MorphAnalyzer

class VerbGovernmentExtractor():

    
    def __init__(self, window_size=50_000_000):

        # Инициализируем пути и секретные коды.
        if pathlib.Path('gov_processing.ini').is_file():
            config = configparser.ConfigParser()
            config.read('gov_processing.ini', encoding="utf-8")
            for sect in config.sections():
                for key, val in config[sect].items():
                    self.__dict__[key.upper()] = val
        else:
            print('Отсутствует файл инициализации gov_processing.ini. Это вызовет ошибки в работе программы.')
            return

        # Инициализируем логгер
        LOGGING_FORMAT = "%(asctime)s : %(levelname)s : %(message)s"
        logging.basicConfig(format=LOGGING_FORMAT, level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Инициализирован логгер")

        # Подключаем Минио
        self.client = minio.Minio(
            self.ADDRESS, access_key=self.ACCESS_KEY, secret_key=self.SECRET_KEY, secure=False
        )

        # Задаем размер окна для побайтового чтения файлов из Минио
        self.WINDOW_SIZE = window_size

        # Задаем директорию, куда будут записываться результаты
        if not pathlib.Path(self.RESULTS_PATH).exists():
            pathlib.Path(self.RESULTS_PATH).mkdir(parents=True, exist_ok=True)
        self.INDEX_TABLE_PATH = f"{self.RESULTS_PATH}/_index_table.csv"

        # Инициализируем pymorphy
        self.pymorphy = MorphAnalyzer()
        
        # Загружаем файл с индексами conllu-файлов
        if os.path.exists(self.FILE2ID_PATH):
            with open(self.FILE2ID_PATH, "r", encoding="utf-8") as file:
                self.file2id = json.load(file)
            self.logger.info("Загружен файл с индексами conllu-файлов")
            self.id2file = {v: k for k, v in self.file2id.items()}
        else:
            self.logger.error(f"Не найден файл с индексами conllu-файлов `{self.FILE2ID_PATH}`")
            exit()

        # Загружаем словарь предложного управления
        if os.path.exists(self.PREPOSITIONAL_PATH):
            with open(self.PREPOSITIONAL_PATH, "r", encoding="utf-8") as file:
                self.prepositional_government = json.load(file)
            self.logger.info("Загружен словарь предложного управления")
        else:
            self.logger.error(f"Не найден файл со словарем предложного управления `{self.PREPOSITIONAL_PATH}`")
            exit()

        # Загружаем список вводных конструкций
        if os.path.exists(self.PARENTHETICAL_PATH):
            with open(self.PARENTHETICAL_PATH, "r", encoding="utf-8") as file:
                self.parenthetical = json.load(file)
            self.logger.info("Загружен список вводных конструкций")
        else:
            self.logger.error(f"Не найден файл со списком вводных конструкций `{self.PARENTHETICAL_PATH}`")
            exit()

    def init_statistics(self):
        """Инициализирует словарь, куда сохраняется статистика."""
        return {
            "verbs_initial": 0,
            "verbs_without_government": 0,
            "verbs_xad": 0,
            "verbs_flexion": 0,
            "verbs_symbols": 0,
            "verbs_yo": 0,
            "verbs_pymorphy": 0,
            "verbs_opencorpora": 0,
            "nouns_initial": 0,
            "nouns_xad": 0,
            "nouns_yo": 0,
            "nouns_opencorpora": 0,
            "nouns_pymorphy": 0,
            "prepositions_initial": 0,
            "prepositions_xad": 0,
            "prepositions_yo": 0,
            "prepositions_opencorpora": 0,
            "parenthetical": 0,
            "prepositional_government": 0
        }
      
    def is_quantifier(self, token):
        """Проверяет, является ли токен квантификатором"""
        # advs - необязательный пункт, т.к. квантификатор обычно зависит от глагола, а не от существительного
        advs = [
            "много", "немного", "сколько", "несколько",
            "больше", "меньше", "достаточно", "столько",
            "мало", "немало", "много", "немного",
            "чуть-чуть", "чуть"
        ]
        try:
            if token["upos"] == "NUM" or token["form"] in advs:
                return True 
            else:
                return False 
        except KeyError:
            return False

    def is_in_opencorpora(self, lemma, preposition=False):
        """Проверяет, входит ли лемма в словарь opencorpora."""

        def check(lemma):
            norm = []
            parsed = [i for i in self.pymorphy.parse(lemma)]
            for i in parsed:
                if type(i.methods_stack[0][0]) == pymorphy2.units.by_lookup.DictionaryAnalyzer:
                    if len(i.methods_stack) > 1:
                        if type(i.methods_stack[1][0]) != pymorphy2.units.by_analogy.UnknownPrefixAnalyzer:
                            norm.append(i)
                    else:
                        norm.append(i)
            return bool(norm)
        
        if preposition:
            split_preposition = lemma.split()
            norm_preposition = []
            for part in split_preposition:
                if part == "NO":
                    norm_preposition.append(part)
                else:
                    if check(part):
                        norm_preposition.append(part)
            if len(norm_preposition) != len(split_preposition):
                return False
            else:
                return True
        else:
            return check(lemma)

    def is_parenthetical(self, preposition, noun, noun_children):   
        """Проверяет, является ли комбинация предлога и существительного вводной конструкцией."""     
        phrase = preposition + " " + noun["form"] if preposition != "NO" else noun["form"]
        if phrase in self.parenthetical["preposition_noun"]:
            return True 
        if preposition in self.parenthetical["not_preposition"]:
            return True 
        if phrase in self.parenthetical["determinant"]:
            dets = noun_children.filter(upos="DET")
            if dets:
                return True
        for child in noun_children:
            if preposition + " " + child["form"] + " " + noun["form"] in self.parenthetical["fixed"]:
                return True
        return False

    def get_verb_government(self, tokenlist):
        """Возвращает два словаря из токенлиста: информацию об управлении глагола в данном токенлисте и статистику."""

        statistics = self.init_statistics()

        # ищем глаголы в списке токенов
        verbs = {}
        for token in tokenlist.filter(upos="VERB"):
            statistics["verbs_initial"] += 1
            relevant = True

            # фильтруем глагол
            lemma = token["lemma"]
            if lemma:
                # заменяем xad
                if "\xad" in token["lemma"]:
                    statistics["verbs_xad"] += 1
                    lemma = lemma.replace("\xad", "")
                # заменяем ё
                if "ё" in token["lemma"]:
                    statistics["verbs_yo"] += 1
                    lemma = lemma.replace("ё", "е")
                # игнорируем глаголы с окончаниями прилагательных
                if lemma[-2:] in ["ый", "ий", "ой"]:
                    statistics["verbs_flexion"] += 1
                    relevant = False
                # игнорируем глаголы с некириллическими символами
                if re.findall('[^а-яА-ЯёЁ-]', lemma):
                    statistics["verbs_symbols"] += 1
                    relevant = False
                # игнорируем глаголы, которые pymorphy не определяет как глагол
                if not list(set([i.tag.POS for i in self.pymorphy.parse(lemma)]) & set(['VERB', 'INFN'])):
                    statistics["verbs_pymorphy"] += 1
                    relevant = False
                # игнорируем глаголы, которых нет в Opencorpora
                if not self.is_in_opencorpora(lemma):
                    statistics["verbs_opencorpora"] += 1 
                    relevant = False
            else:
                continue

            # сохраняем id глагола и лемму, если прошел все фильтры
            if relevant:                  
                verbs[token["id"]] = {
                    "lemma": lemma,
                    "preps_cases": [],
                    "nouns": []
                }

        # проходимся по каждому глаголу
        for verb_id in verbs:
            verb_children = tokenlist.filter(head=verb_id)

            # избегаем случаи, когда глагол встретился с отрицанием
            no = verb_children.filter(form=lambda x: x.lower() == "не")
            if no:
                continue

            # проходимся по каждому зависимому существительному
            nouns = verb_children.filter(upos=lambda x: x in ["NOUN", "PROPN"], deprel=lambda x: x != "parataxis")
            for noun in nouns:
            
                noun_children = tokenlist.filter(head=noun["id"])
                statistics["nouns_initial"] += 1

                # пропускаем существительные, если для них не определен падеж
                case = noun["feats"]["Case"]
                if not case:
                    continue

                # фильтруем существительное
                noun_lemma = noun["lemma"]
                if noun_lemma:
                    relevant = True
                    if "\xad" in noun["lemma"]:
                        statistics["nouns_xad"] += 1
                        noun_lemma = noun_lemma.replace("\xad", "")
                    if "ё" in noun["lemma"]:
                        statistics["nouns_yo"] += 1
                        noun_lemma = noun_lemma.replace("ё", "е")
                    if not self.is_in_opencorpora(noun_lemma):
                        statistics["nouns_opencorpora"] += 1
                        relevant = False
                    if not list(set([i.tag.POS for i in self.pymorphy.parse(noun_lemma)]) & set(['NOUN', 'NPRO'])):
                        statistics["nouns_pymorphy"] += 1
                        relevant = False
                    if not relevant:
                        continue 

                    else:
                        # игнорируем те существительные, с которыми в связке идут квантификаторы
                        quantifiers_count = 0
                        for noun_child in noun_children:
                            if self.is_quantifier(noun_child):
                                quantifiers_count += 1
                        if quantifiers_count:
                            continue
                        
                        # проверяем наличие предлога, при наличии сохраняем, при отсутствии определяем его как NO
                        prepositions = []
                        adps = noun_children.filter(upos="ADP", deprel="case")
                        for adp in adps:
                            prep = adp["form"].lower()
                            adp_fixed = tokenlist.filter(head=adp["id"], deprel="fixed")
                            for fixed in adp_fixed:
                                prep += " " + fixed["form"].lower()
                            prepositions.append(prep.strip())
                        if prepositions:
                            preposition = prepositions[0]
                            statistics["prepositions_initial"] += 1
                            # фильтруем предлог
                            if "\xad" in preposition:
                                statistics["prepositions_xad"] += 1
                                preposition = preposition.replace("\xad", "")
                            if "ё" in preposition:
                                statistics["prepositions_yo"] += 1 
                                preposition = preposition.replace("ё", "е")
                            if preposition.split()[-1] in [
                                "со", "ко", "во", "обо", "безо", "надо", "подо", "ото", "передо", "изо", "предо"
                            ]:
                                preposition = preposition[:-1]
                            if not self.is_in_opencorpora(preposition, preposition=True):
                                statistics["prepositions_opencorpora"] += 1 
                                continue
                        else:
                            preposition = "NO"

                        # фильтруем вводные слова
                        if self.is_parenthetical(preposition, noun, noun_children):
                            statistics["parenthetical"] += 1
                            continue 
                        else:
                            # сохраняем результат
                            verbs[verb_id]["preps_cases"].append(preposition + "_" + case)
                            verbs[verb_id]["nouns"].append(noun_lemma)

                else:
                    continue

        verbs_final = {}
        for verb_id, verb_info in verbs.items():
            if verb_info["preps_cases"] and verb_info["nouns"]:
                # фильтруем по словарю предложного управления
                wrong_case = 0
                for prep_case in verb_info["preps_cases"]:
                    prep = prep_case.split("_")[0]
                    case = prep_case.split("_")[1]
                    if prep in self.prepositional_government:
                        if case not in self.prepositional_government[prep]:
                            statistics["prepositional_government"] += 1
                            wrong_case += 1
                if wrong_case == 0:
                    verbs_final[verb_id] = verb_info
            else:
                statistics["verbs_without_government"] += 1

        return verbs_final, statistics

    def add_statistics(self, initial_statistics, added_statistics):
        """Объединяет два файла со статистикой."""
        return {k: initial_statistics.get(k, 0) + added_statistics.get(k, 0) for k in set(initial_statistics)}

    def add_government(self, initial_government, added_government, number_of_item):
        """Объединяет два файла с управлением."""

        number_of_item = str(number_of_item)

        for verb in added_government:
            # если глагола нет в изначальном словаре, копируем информацию об этом глаголе из нового словаря в изначальный
            # и добавляем информацию о номере элемента (чанка или файла) в словарь
            if verb not in initial_government:
                initial_government[verb] = added_government[verb]
                for prep_case in initial_government[verb]:
                    for nouns in initial_government[verb][prep_case]:
                        initial_government[verb][prep_case][nouns][1] = number_of_item + "-" + initial_government[verb][prep_case][nouns][1] 
            # если глагол есть в изначальном словаре               
            else:
                # для каждой связки "предлог+падеж" ищем такую же связку для этого глагола в изначальном словаре
                for prep_case_added in added_government[verb]:
                    prep_case_found = None
                    for prep_case_initial in initial_government[verb]:
                        if Counter(prep_case_added) == Counter(prep_case_initial):
                            prep_case_found = prep_case_initial
                            break
                    # если такая же связка нашлась
                    if prep_case_found:
                        # для каждой группы существительных ищем такие же существительные для этой связки в изначальном словаре
                        for nouns_added in added_government[verb][prep_case_added]:
                            nouns_found = False
                            added_zipped = [(i, j) for i, j in zip(prep_case_added, nouns_added.split(","))]
                            for nouns_initial in initial_government[verb][prep_case_found]:
                                initial_zipped = [(i, j) for i, j in zip(prep_case_found, nouns_initial.split(","))]
                                if set(initial_zipped) == set(added_zipped):
                                    nouns_found = True
                                    break 
                            # если такая группа существительных нашлась, добавляем count к count в изначальном словаре
                            if nouns_found:
                                initial_government[verb][prep_case_found][nouns_initial][0] += added_government[verb][prep_case_added][nouns_added][0]
                            # если такая группа существительных не нашлась, добавляем ее в изначальный словарь
                            # и к этой группе добавляем информацию о номере элемента (чанка или файла)
                            else:
                                prep_case_added_list = list(prep_case_added)
                                nouns_to_add = []
                                for prep_case in prep_case_found:
                                    nouns_to_add.append(nouns_added.split(",")[prep_case_added_list.index(prep_case)])
                                    prep_case_added_list[prep_case_added_list.index(prep_case)] = ""
                                nouns_to_add = ",".join(nouns_to_add)                          
                                initial_government[verb][prep_case_found][nouns_to_add] = added_government[verb][prep_case_added][nouns_added]
                                initial_government[verb][prep_case_found][nouns_to_add][1] = number_of_item + "-" + added_government[verb][prep_case_added][nouns_added][1]
                    # если такой связки нет в изначальном словаре, добавляем ее в изначальный словарь
                    # и добавляем информацию о номере элемента (чанке или файле)
                    else:
                        initial_government[verb][prep_case_added] = added_government[verb][prep_case_added]
                        for nouns in initial_government[verb][prep_case_added]:
                            initial_government[verb][prep_case_added][nouns][1] = number_of_item + "-" + initial_government[verb][prep_case_added][nouns][1] 

        return initial_government    

    def read_next_chunk(self, conllu_file, chunk_start, window_size):
        """Читает побайтово следующий блок данных из текущего файла."""
        decoded = ""
        chunk_end = chunk_start + window_size
        data_bytes = self.client.get_object(
            self.BUCKET_NAME, "syntax-parsed/"+conllu_file, offset=chunk_start, length=self.WINDOW_SIZE
        ).data
        try:
            data_bytes.decode("utf-8")
        except UnicodeDecodeError:
            for i in range(4):
                try:
                    data_bytes = self.client.get_object(
                        self.BUCKET_NAME, "syntax-parsed/"+conllu_file, offset=chunk_start, length=self.WINDOW_SIZE + i
                    ).data
                    data_bytes.decode("utf-8")
                    break
                except UnicodeDecodeError:
                    continue
        data_bytes_reversed = data_bytes[::-1]
        for i, byte in enumerate(data_bytes_reversed[:-4]):
            prev_bytes = [data_bytes_reversed[i+j] for j in range(1, 4)]
            try:
                if byte == 10 and prev_bytes[0] == 13 and prev_bytes[1] == 10 and prev_bytes[2] == 13:
                    if i == 0:
                        decoded = data_bytes.decode("utf-8")
                        chunk_end = chunk_start + len(data_bytes)
                    else:
                        decoded = data_bytes[:-i].decode("utf-8")
                        chunk_end = chunk_start + len(data_bytes[:-i])
                    break
                if byte == 10 and prev_bytes[0] == 10:
                    if i == 0:
                        decoded = data_bytes.decode("utf-8")
                        chunk_end = chunk_start + len(data_bytes)
                    else:
                        decoded = data_bytes[:-i].decode("utf-8")
                        chunk_end = chunk_start + len(data_bytes[:-i])
                    break
            except UnicodeDecodeError:
                continue
        return decoded, chunk_start, chunk_end
    
    def process_chunk(self, chunk):
        """
        Обрабатывает блок данных в текущем файле. 
        Возвращает два словаря: информацию об управлении глагола в данном блоке и статистику
        """
        statistics_chunk = self.init_statistics()
        government_chunk = {}
        chunk = chunk.strip()
        texts = chunk.split("\r\n\r\n")
        if len(texts) == 1:
            texts = chunk.split("\n\n")
        line = 0
        for text in texts:
            lines = len(text.split("\n"))
            try:
                tokenlist = conllu.parse(text)
                verbs_tokenlist, statistics_tokenlist = self.get_verb_government(tokenlist[0])
                statistics_chunk = self.add_statistics(statistics_chunk, statistics_tokenlist)
                if verbs_tokenlist:
                    for verb_info in verbs_tokenlist.values():
                        if verb_info["lemma"] not in government_chunk:
                            government_chunk[verb_info["lemma"]] = {
                                    tuple(verb_info["preps_cases"]): {
                                        ",".join(verb_info["nouns"]): [1, str(line)]
                                    }
                            }
                        else:
                            prep_case_found = None
                            for prep_case in government_chunk[verb_info["lemma"]]:
                                if set(prep_case) == set(verb_info["preps_cases"]):
                                    prep_case_found = prep_case
                                    break
                            if prep_case_found:
                                nouns_transformed = ",".join([verb_info["nouns"][verb_info["preps_cases"].index(prep_case)] for prep_case in prep_case_found])
                                nouns_found = False
                                for nouns in government_chunk[verb_info["lemma"]][prep_case_found]:
                                    if nouns == nouns_transformed:
                                        nouns_found = True
                                        government_chunk[verb_info["lemma"]][prep_case_found][nouns_transformed][0] += 1
                                        break
                                if not nouns_found:
                                    government_chunk[verb_info["lemma"]][prep_case_found][nouns_transformed] = [1, str(line)]
                            else:
                                government_chunk[verb_info["lemma"]][tuple(verb_info["preps_cases"])] = {
                                    ",".join(verb_info["nouns"]): [1, str(line)]
                                }
                line += lines + 1
            except:
                line += lines + 1
                continue
        return government_chunk, statistics_chunk

    def process_file(self, conllu_file, file_id, save_every, locki):
        """
        Обрабатывает файл.
        При обработке каждые `save_every` блоков файла (чанков) записывает результат.
        """

        self.logger.info(f"{file_id} : {conllu_file} : начинается обработка файла")

        file_size = self.client.stat_object(self.BUCKET_NAME, "syntax-parsed/"+conllu_file).size
        to_index_table = ""
        chunk_id = 0
        chunk_start = 0
        government_file = {}
        statistics_file = self.init_statistics()

        if os.path.exists(f"{self.INDEX_TABLE_PATH}"):
            index_table = pd.read_csv(f"{self.INDEX_TABLE_PATH}")
            chunk_id = index_table[index_table.FileId == file_id].chunk_id.max()
            chunk_end = index_table[index_table.FileId == file_id].chunk_end.max()
            if pd.isnull(chunk_id) and pd.isnull(chunk_end):
                chunk_id = 0 
            else:
                with open(f"{self.RESULTS_PATH}/{file_id}_government.pickle", "rb") as file:
                    government_file = pickle.load(file)
                with open(f"{self.RESULTS_PATH}/{file_id}_statistics.json", "r", encoding="utf-8") as file:
                    statistics_file = json.load(file)
                chunk_id += 1
                chunk_start = chunk_end
                file_size -= chunk_end
        else:
            with open(f"{self.INDEX_TABLE_PATH}", "w", encoding="utf-8") as file:
                file.write("Filename,FileId,ChunkId,ChunkStart,ChunkEnd\n")
        
        chunk, chunk_start, chunk_end = self.read_next_chunk(
            conllu_file=conllu_file, chunk_start=chunk_start, window_size=self.WINDOW_SIZE
        )
        government_chunk, statistics_chunk = self.process_chunk(chunk)
        government_file = self.add_government(government_file, government_chunk, chunk_id)
        statistics_file = self.add_statistics(statistics_file, statistics_chunk)
        to_index_table += f"{conllu_file},{file_id},{chunk_id},{chunk_start},{chunk_end}\n"
        chunk_id += 1

        n_chunks = math.ceil(file_size / self.WINDOW_SIZE) - 1

        if n_chunks == 0:
            self.logger.info(f"{file_id} : {conllu_file} : 1 из {n_chunks + 1} : запись результатов")
            with open(f"{self.RESULTS_PATH}/{file_id}_government.pickle", "wb") as file:
                pickle.dump(government_file, file)
            with open(f"{self.RESULTS_PATH}/{file_id}_statistics.json", "w") as file:
                json.dump(statistics_file, file, ensure_ascii=False, indent=2)
            locki.acquire()
            with open(f"{self.INDEX_TABLE_PATH}", "a", encoding="utf-8") as file:
                file.write(to_index_table)
            locki.release()
            to_index_table = ""
            self.logger.info(f"{file_id} : {conllu_file} : {n_chunks + 1} : запись результатов завершена")

        else:        
            for i in range(1, n_chunks + 1):
                try:
                    chunk, chunk_start, chunk_end = self.read_next_chunk(
                        conllu_file=conllu_file, chunk_start=chunk_end, window_size=self.WINDOW_SIZE
                    )
                    government_chunk, statistics_chunk = self.process_chunk(chunk)
                    government_file = self.add_government(government_file, government_chunk, chunk_id)
                    statistics_file = self.add_statistics(statistics_file, statistics_chunk)
                    to_index_table += f"{conllu_file},{file_id},{chunk_id},{chunk_start},{chunk_end}\n"
                    chunk_id += 1      
                    if i % save_every == 0 or i == n_chunks:
                        self.logger.info(f"{file_id} : {conllu_file} : {i} из {n_chunks} : запись результатов")
                        with open(f"{self.RESULTS_PATH}/{file_id}_government.pickle", "wb") as file:
                            pickle.dump(government_file, file)
                        with open(f"{self.RESULTS_PATH}/{file_id}_statistics.json", "w") as file:
                            json.dump(statistics_file, file, ensure_ascii=False, indent=2)
                        locki.acquire()
                        with open(f"{self.INDEX_TABLE_PATH}", "a", encoding="utf-8") as file:
                            file.write(to_index_table)
                        locki.release()
                        to_index_table = ""
                        self.logger.info(f"{file_id} : {conllu_file} : {i} из {n_chunks} : запись результатов завершена")
                except minio.error.S3Error:
                    break

    def process_files(self, conllu_files, save_every=5):
        """Основная функция обработки файлов."""

        files_not_in_file2id = [conllu_file for conllu_file in conllu_files if conllu_file not in self.file2id]
        if files_not_in_file2id:
            self.logger.error(f"Для следующих conllu-файлов не указан индекс в {self.FILE2ID_PATH}: {' '.join(files_not_in_file2id)}")
            exit()
        else:
            self.logger.info(f"Для всех conllu-файлов успешно найден индекс в {self.FILE2ID_PATH}")

        locki = threading.Lock()
        with ThreadPoolExecutor(max_workers=5) as executor:
            for conllu_file in conllu_files:
                file_id = self.file2id[conllu_file]
                executor.submit(self.process_file, conllu_file, file_id, save_every=save_every, locki=locki)

    def join_files_statistics(self, files):
        """Объединяет статистики по указанным файлам в один общий файл."""

        if os.path.exists(f"{self.RESULTS_PATH}/ALL_statistics.json"):
            with open(f"{self.RESULTS_PATH}/ALL_statistics.json", "r", encoding="utf-8") as file:
                statistics = json.load(file)
        else:
            statistics = self.init_statistics()

        self.logger.info("Начинается объединение статистики")

        for file in tqdm(files):
            file_id = self.file2id[file]
            with open(f"{self.RESULTS_PATH}/{file_id}_statistics.json", "r", encoding="utf-8") as file:
                statistics_file = json.load(file)
            statistics = self.add_statistics(statistics, statistics_file)

        with open(f"{self.RESULTS_PATH}/ALL_statistics.json", "w", encoding="utf-8") as file:
            json.dump(statistics, file, ensure_ascii=False, indent=2)

        self.logger.info(f"Объединение статистики завершено и сохранено в: {self.RESULTS_PATH}/ALL_statistics.json")
    
    def join_files_government(self, files):
        """Объединяет управление из указанных файлов в один общий файл."""

        if os.path.exists(f"{self.RESULTS_PATH}/ALL_government.pickle"):
            with open("{self.RESULTS_PATH}/ALL_government.pickle", "rb") as file:
                government = pickle.load(file)
        else:
            government = {}

        self.logger.info("Начинается объединение глагольного управления")

        for file in tqdm(files):
            file_id = self.file2id[file]
            with open(f"{self.RESULTS_PATH}/{file_id}_government.pickle", "rb") as file:
                government_file = pickle.load(file)
            government = self.add_government(government, government_file, file_id)
        
        self.logger.info("Запись объединенного глагольного управления в файл...")
        with open(f"{self.RESULTS_PATH}/ALL_government.pickle", "wb") as file:
            pickle.dump(government, file)
        self.logger.info(f"Объединение глагольного управления завершено и сохранено в: {self.RESULTS_PATH}/ALL_government.pickle")

    def find_example(self, idx: str):
        """Печатает пример текста по индексу."""

        idx = idx.split("-")
        file_id = idx[0]
        chunk_id = int(idx[1])
        line_id = int(idx[2])

        index_table = pd.read_csv(self.INDEX_TABLE_PATH)
        chunk_start = index_table[(index_table.FileId == file_id) & (index_table.ChunkId == chunk_id)].ChunkStart.values[0]
        chunk_end = index_table[(index_table.FileId == file_id) & (index_table.ChunkId == chunk_id)].ChunkEnd.values[0]

        chunk = self.client.get_object(
                        self.BUCKET_NAME, "syntax-parsed/"+self.id2file[file_id], offset=chunk_start, length=chunk_end-chunk_start
                    ).data
        
        lines = chunk.decode("utf-8").split("\n")

        for i, line in enumerate(lines[line_id:]):
            if not lines[line_id:][i + 1].startswith("#"):
                print(line)
            else:
                print(line)
                break
    
if __name__ == "__main__":  

    files = []

    # параметры VerbGovernmentExtractor можно менять:
    # window_size: размер окна в байтах при чтении conllu-файлов из Minio (по умолчанию: 50_000_000)
    # results_path: наименование директории, куда будут складываться результаты обработки (по умолчанию: "results")
    extractor = VerbGovernmentExtractor()

    # save_every: количество блоков файла, после обработки которых результаты сохраняются в results_path
    # extractor.process_files(files, save_every=2)

    # после обработки всех файлов статистику и управление можно объединить
    files = list(extractor.file2id.keys())
    extractor.join_files_statistics(files)
    extractor.join_files_government(files)

