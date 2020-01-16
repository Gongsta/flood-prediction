
from eolearn.core import EOPatch

from eolearn.core import AddFeature  # AddFeature is a simple EOTask which adds a feature to a given EOPatch

patch = EOPatch()

import numpy as np
from eolearn.core import FeatureType

new_bands = np.zeros((5, 10, 10, 13), dtype=np.float32)

patch[FeatureType.DATA]['bands'] = new_bands
# or patch.data['bands'] = new_bands

feature = (FeatureType.DATA, 'bands')
add_feature = AddFeature(feature)

data = np.zeros((5, 100, 100, 13))
patch = add_feature.execute(patch, data)
# or patch = add_feature(patch, data)

from eolearn.core import CopyTask, RenameFeature

copy_task = CopyTask()
rename_feature = RenameFeature((FeatureType.DATA, 'bands', 'the_bands'))
copy_rename_task = rename_feature * copy_task

new_patch = copy_rename_task(patch)
new_patch

from eolearn.core import EOWorkflow, Dependency

workflow = EOWorkflow([
    Dependency(add_feature, inputs=[]),
    Dependency(copy_task, inputs=[add_feature]),
    Dependency(rename_feature, inputs=[copy_task])
])
# Instead of Dependecy class also just a tuple can be used

result = workflow.execute({
    add_feature: {'eopatch': patch,
                  'data': new_bands}
})

result

from eolearn.core import LinearWorkflow

workflow = LinearWorkflow(add_feature, copy_task, rename_feature)

result = workflow.execute({
    add_feature: {'eopatch': patch,
                  'data': new_bands}
})


workflow.dependency_graph()