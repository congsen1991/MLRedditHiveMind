# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

A = 'daklk,djkal,jkdajoijfe.jdlsajo daf'

B = ''
for char in A:
    if char.isalpha():
        B+=char
    else: B+= ' '

print B.split()