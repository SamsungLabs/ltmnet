"""
Copyright (c) 2022 Samsung Electronics Co., Ltd.

Author(s):
Luxi Zhao (lucy.zhao@samsung.com; lucyzhao.zlx@gmail.com)

Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-sa/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""

"""
Execution command:
cd ~/LTMNet; python3 -m prepare.analyze_mit_adobe.run_analyze_mit_adobe
"""

import os


def run_analysis(dataset_dir):
    args_str = '--input_dir "{ds}/input" ' \
               '--gt_dir "{ds}/gt" ' \
               '--out_dir "./outputs/mit-adobe-fivek-transfer-functions-expertcwb-expertc-all-deg4" ' \
               '--poly_order 4 ' \
               '--savefig ' \
        .format(ds=dataset_dir)
    print('Starting analysis...')
    print('args_str = {}'.format(args_str))
    os.system('python3 -m prepare.analyze_mit_adobe.analyze_mit_adobe {} '.format(args_str))


if __name__ == '__main__':
    os.system('pwd')
    os.system('pip install sklearn')
    dataset_dir = '/home/user/Data'
    run_analysis(dataset_dir)
