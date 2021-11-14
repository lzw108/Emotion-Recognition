import re


def clean_text(htmlstr):
    # 过滤DOCTYPE
    htmlstr = ' '.join(htmlstr.split())  # 去掉多余的空格
    re_doctype = re.compile(r'<!DOCTYPE .*?> ', re.S)
    s = re_doctype.sub('', htmlstr)
    # 过滤CDATA
    re_cdata = re.compile('//<!CDATA\[[ >]∗ //\] > ', re.I)
    s = re_cdata.sub('', s)
    # Script
    re_script = re.compile('<\s*script[^>]*>[^<]*<\s*/\s*script\s*>', re.I)
    s = re_script.sub('', s)  # 去掉SCRIPT
    # style
    re_style = re.compile('<\s*style[^>]*>[^<]*<\s*/\s*style\s*>', re.I)
    s = re_style.sub('', s)  # 去掉style
    # 处理换行
    re_br = re.compile('<br\s*?/?>')
    s = re_br.sub('', s)  # 将br转换为换行
    # HTML标签
    re_h = re.compile('</?\w+[^>]*>')
    s = re_h.sub('', s)  # 去掉HTML 标签
    # HTML注释
    re_comment = re.compile('<!--[^>]*-->')
    s = re_comment.sub('', s)
    # 多余的空行
    blank_line = re.compile('\n+')
    s = blank_line.sub('', s)
    # 剔除超链接
    http_link = re.compile(r'(http://.+.html)')
    s = http_link.sub('', s)
    return s


def process_text(df):
    df['微博中文内容'] = df['微博中文内容'].apply(lambda x: clean_text(x))
    return df
