from findhr.preprocess.metadata import JSONMetadata

# Define the metadata for the JDS dataset
md_JDS = {
    'qId': JSONMetadata(schema={'type': 'number'}),
    'Occupation_j': JSONMetadata(schema={'type': 'string'}),
    'Education_j': JSONMetadata(schema={'enum': ['No education', 'Degree', 'Bachelor D.', 'Master D.', 'PhD', 'Any']},
                              attr_type='category'),
    'Age_j': JSONMetadata(schema={'type': 'array',
                                  'prefixItems': [
                                    { 'type': 'number' },
                                    { 'type': 'number' },
                                  ],
                                  'items': False},
                          ),
    'Gender_j': JSONMetadata(schema={'enum': ['Male', 'Female', 'Non-binary', 'Any']},
                             attr_type='category', attr_usage='sensitive'),
    'Contract_j': JSONMetadata(schema={'enum': ['Remote', 'Hybrid', 'In presence']}),
    'Nationality_j': JSONMetadata(schema={'type': 'string'}),
    'Competences_j': JSONMetadata(schema={'type': "array", 'items': {'type': 'string'}}),
    'Knowledge_j': JSONMetadata(schema={'type': "array", 'items': {'type': 'string'} }),
    'Languages_j': JSONMetadata(schema={'type': "array", 'items': {'type': 'string'}}),
    'Experience_j': JSONMetadata(schema={'type': 'number'}),
}

# Define the metadata for the CDS dataset
md_CDS = {
    'kId': JSONMetadata(schema={'type': 'integer'}),
    'Occupation_c': JSONMetadata(schema={'type': 'string'}),
    'Education_c': JSONMetadata(schema={'enum': ['No education', 'Degree', 'Bachelor D.', 'Master D.', 'PhD', 'Any']},
                              attr_type='category'),
    'Age_c': JSONMetadata(schema={'type': 'number'}),
    'Gender_c': JSONMetadata(schema={'enum': ['Male', 'Female', 'Non-binary']},
                             attr_type='category', attr_usage='sensitive'),
    'Contract_c': JSONMetadata(schema={'enum': ['Remote', 'Hybrid', 'In presence', 'Any']}, attr_type='category'),
    'Nationality_c': JSONMetadata(schema={'type': 'string'}),
    'Competences_c': JSONMetadata(schema={'type': "array", 'items': {'type': 'string'}}),
    'Knowledge_c': JSONMetadata(schema={'type': "array", 'items': {'type': 'string'}}),
    'Experience_c': JSONMetadata(schema={'type': 'number'}),
    'Languages_c': JSONMetadata(schema={'type': "array",'items': {'type': 'string'}}),
}

md_ADS = {
    'rank': JSONMetadata(schema={'type': 'number', 'attr_usage':'target'}),
    'score': JSONMetadata(schema={'type': 'number', 'attr_usage':'target'}),
}
md_CDS_JDS_ADS = {**md_CDS, **md_JDS, **md_ADS}
