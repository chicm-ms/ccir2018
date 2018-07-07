import os
import settings

def scan(filename):
    with open(filename, 'r', encoding='UTF-8') as f:
        count = 0
        for line in f:
            if count < 10:
                print(count)
                print(line)
            else:
                break
            count += 1

#scan(r'F:\competition\training_set.txt')
#scan(os.path.join(settings.DATA_DIR, 'user_infos.txt'))
scan(os.path.join(settings.TEST_DATA_DIR, 'testing_set_135089.txt'))