"""
Copyright (c) 2022 Samsung Electronics Co., Ltd.

Author(s):
Abdelrahman Abdelhamed (abdoukamel@gmail.com)

Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-sa/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""

class ExifFormat:
    def __init__(self, id, name, size, short_name):
        self.id = id
        self.name = name
        self.size = size
        self.short_name = short_name  # used with struct.unpack()


exif_formats = {
    1: ExifFormat(1, 'unsigned byte', 1, 'B'),
    2: ExifFormat(2, 'ascii string', 1, 's'),
    3: ExifFormat(3, 'unsigned short', 2, 'H'),
    4: ExifFormat(4, 'unsigned long', 4, 'L'),
    5: ExifFormat(5, 'unsigned rational', 8, ''),
    6: ExifFormat(6, 'signed byte', 1, 'b'),
    7: ExifFormat(7, 'undefined', 1, 'B'),  # consider `undefined` as `unsigned byte`
    8: ExifFormat(8, 'signed short', 2, 'h'),
    9: ExifFormat(9, 'signed long', 4, 'l'),
    10: ExifFormat(10, 'signed rational', 8, ''),
    11: ExifFormat(11, 'single float', 4, 'f'),
    12: ExifFormat(12, 'double float', 8, 'd'),
}