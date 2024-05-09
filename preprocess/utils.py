import os
from glob import glob

from slugify import slugify
from tqdm import tqdm
import json
import pickle


def is_hex(s):
    try:
        int(s, 16)
        return True
    except ValueError:
        return False

def extract_emails_and_passwords(txt_lines):
    emails_passwords = []
    for txt_line in txt_lines:
        try:
            if ':' in txt_line and '@' in txt_line:
                strip_txt_line = txt_line.strip()
                email, password = strip_txt_line.split(':')
                if all(32 <= ord(password[i]) <= 126 for i in range(len(password))) and (' ' not in password) and (len(password)>=4) and (len(password)<=30) and (not (is_hex(password) and len(password)>=20)):
                    if ('@' in email) and (' ' not in email) and all(32 <= ord(email[i]) <= 126 for i in range(len(email))):
                        emails_passwords.append((email, password))
        except:
            pass
    return emails_passwords


def read_all(breach_compilation_folder, on_file_read_call_back):
    read_n_files(breach_compilation_folder, None, on_file_read_call_back)


def read_n_files(breach_compilation_folder, num_files, on_file_read_call_back_class):
    breach_compilation_folder = os.path.join(os.path.expanduser(breach_compilation_folder), 'data')
    all_filenames = glob(breach_compilation_folder + '/**/*', recursive=True)
    callback_class_name = str(on_file_read_call_back_class).split('callback.')[-1][:-2]
    output_dir = os.path.join(os.path.expanduser('~/BreachCompilationAnalysis'), callback_class_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print('Found {0} files'.format(len(all_filenames)))
    if num_files is not None:
        all_filenames = all_filenames[0:num_files]
    for current_filename in tqdm(all_filenames):
        if os.path.isfile(current_filename):
            suffix = slugify(current_filename.split('data')[-1])
            output_filename = os.path.join(output_dir, suffix)
            callback = on_file_read_call_back_class(output_filename)
            with open(current_filename, 'r', encoding='utf8', errors='ignore') as r:
                lines = r.readlines()
                emails_passwords = extract_emails_and_passwords(lines)
                callback.call(emails_passwords)
            callback.persist()


def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


def read_files(breach_compilation_folder):
    breach_compilation_folder = os.path.join(os.path.expanduser(breach_compilation_folder), 'data')
    all_filenames = glob(breach_compilation_folder + '/**/*', recursive=True)
    output_dir = os.path.join(os.path.expanduser('~/BreachCompilationAnalysis'), 'PasswordPairs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print('Found {0} files'.format(len(all_filenames)))
    data_dir = {}
    num = 0
    index = 1
    for current_filename in tqdm(all_filenames):
        if os.path.isfile(current_filename):
            with open(current_filename, 'r', encoding='utf8', errors='ignore') as r:
                lines = r.readlines()
                emails_passwords = extract_emails_and_passwords(lines)
                for (email, password) in emails_passwords:
                    user_name = email.split('@')[0]
                    if user_name not in data_dir.keys():
                        data_dir[user_name] = {}
                    if email not in data_dir[user_name].keys():
                        data_dir[user_name][email] = []
                    data_dir[user_name][email].append(password)
        num = num + 1
        if num % 20 == 0:
            print('finished {} files'.format(num))
            delete_keys = []
            for i in data_dir.keys():
                if (len(data_dir[i].keys()) < 2 and len(data_dir[i][list(data_dir[i].keys())[0]]) < 2) or len(data_dir[i]) > 100:
                # if len(data_dir[i]) > 100:
                    delete_keys.append(i)
            for i in delete_keys:
                del data_dir[i]
            with open(output_dir + '/data_' + str(index) + '.pickle', 'wb') as w:
                pickle.dump(data_dir, w)
            print('saved.')
            index = index + 1
            data_dir = {}
    delete_keys = []
    for i in data_dir.keys():
        if (len(data_dir[i].keys()) < 2 and len(data_dir[i][list(data_dir[i].keys())[0]]) < 2) or len(data_dir[i]) > 100:
        # if len(data_dir[i]) > 100:
            delete_keys.append(i)
    for i in delete_keys:
        del data_dir[i]
    with open(output_dir + '/data_' + str(index) + '.pickle', 'wb') as w:
        pickle.dump(data_dir, w)
    print('done!')
