#!/nfs/user/pag227/miniconda/bin/python
'''Takes a parameters file params.ini and edits it to generate
the luminosity cut views of the data we need for the pairwise
estimator.

Output files will be written in the current directory.'''

from iskay import paramTools

params = paramTools.params('params.ini')
c = params.config

lums_e10 = [4.3, 6.1, 7.9]  # luminosity cuts
resampling = ['jk', 'bs_dt']
constant_query = params.CAT_QUERY

# joint bins
basename = c.get('Name', 'analysisname')
analysis_names1 = [('%s_lum_gt_%04.1f' % (basename, lums)).replace('.',
                                                                   'p')
                   for lums in lums_e10]
fnames_out1 = [('params_lum_gt_%04.1f' % lum).replace('.', 'p')
               for lum in lums_e10]

if constant_query is None:
    queries1 = ['lum > %4.2fe10' % (lum)
                for lum in lums_e10]
else:
    queries1 = ['lum > %4.2fe10 and %s' % (lum, constant_query)
                for lum in lums_e10]

# disjoint bins
analysis_names2 = []
fnames_out2 = []
queries2 = []

for j in range(len(lums_e10)-1):
    lum = lums_e10[j]
    lump1 = lums_e10[j+1]
    name = '%s_disjoint_bin_lum_gt_%04.1f_and_lt_%04.1f' % (basename,
                                                            lum, lump1)
    fname_out = 'params_disjoint_bin_lum_gt_%04.1f_and_%04.1f' % (lum, lump1)
    analysis_names2.append(name.replace('.', 'p'))
    fnames_out2.append(fname_out.replace('.', 'p'))
    if constant_query is None:
        queries2.append('lum > %4.2fe10 and lum < %4.2fe10' % (lum,
                        lump1))  # noqa

    else:
        queries2.append('lum > %4.2fe10 and lum < %4.2fe10 and %s' % (lum,
                        lump1, constant_query))  # noqa

analysis_names = analysis_names1 + analysis_names2
fnames_out = fnames_out1 + fnames_out2
queries = queries1 + queries2

assert len(fnames_out) == len(queries) == len(analysis_names)

print(lums_e10)
print('\n'.join(analysis_names))
print('\n'.join(queries))
print('\n'.join(fnames_out))
# write disjoint bins

for method in resampling:
    for j in range(len(analysis_names)):
        c.set('Name', 'analysisname',
              value=analysis_names[j]+'_' + method)
        c.set('Catalog', 'query', value=queries[j])
        c.set('JK', 'resampling_method', value=method)
        with open(fnames_out[j]+'_%s.ini' % method, 'w') as configfile:
            c.write(configfile)
