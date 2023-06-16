
# ******************************************************************************
# * @attention
# *
# * Copyright (c) 2022 STMicroelectronics.
# * All rights reserved.
# *
# * This software is licensed under terms that can be found in the LICENSE file
# * in the root directory of this software component.
# * If no LICENSE file comes with this software, it is provided AS-IS.
# *
# *
# ******************************************************************************
#

class TypeConversion:

    @staticmethod
    def check_type(check_type):
        switcher = {
            'uint8_t': 'uint8',
            'uint16_t': 'uint16',
            'uint32_t': 'uint32',
            'int8_t': 'int8',
            'int16_t': 'int16',
            'int32_t': 'int32',
            'float': 'float32',
            'double': 'double',
        }
        return switcher.get(check_type, "error")

    @staticmethod
    def check_type_length(check_type):
        switcher = {
            'uint8_t': 1,
            'int8_t': 1,
            'uint16_t': 2,
            'int16_t': 2,
            'uint32_t': 4,
            'int32_t': 4,
            'float': 4,
            'double': 8,
        }
        return switcher.get(check_type, "error")
