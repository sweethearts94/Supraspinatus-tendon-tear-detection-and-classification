import pandas as pd

def read(file_path: str, read_cols: any = None, sheet_name: any = 0, header: int = 0, convert: dict = None) -> pd.DataFrame:
    return pd.read_excel(io=file_path, sheet_name=sheet_name, header=header, usecols=read_cols, converters=convert)

def excel2dict(file_path: str, read_cols: any = None, sheet_name: any = 0, header: int = 0, index_key: str = "", convert: dict = None) -> tuple:
    excel_content = read(file_path, read_cols, sheet_name, header, convert)
    if index_key != "":
        return_list = {}
    else:
        return_list = []
    excel_keys = excel_content.keys()
    for index in range(len(excel_content[excel_keys[0]])):
        single_record = {str(e_k): excel_content[e_k][index] for e_k in excel_keys}
        if index_key == "" or index_key not in excel_keys:
            return_list.append(single_record)
        else:
            return_list[excel_content[index_key][index]] = single_record
    return return_list, list(excel_keys)

if __name__ == "__main__":
    import os
    excel_path = ""
    a, b = excel2dict(os.path.join(*excel_path), read_cols="A,D,H,I,M")
    print(a)
    print(b)