"""
Copyright (C) 2019 University of Massachusetts Amherst.
This file is part of "expLinkage"
http://github.com/iesl/expLinkage
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from pathlib import Path

def fixDataFormat(origDir ,newDir):

    Path(newDir).mkdir(parents=True, exist_ok=True)

    canopyList = sorted([str(f).split("/")[-1] for f in Path(origDir).glob("*") if f.is_dir()])

    print("CanopyList:",canopyList)

    for canopy in canopyList:
        Path("{}/{}".format(newDir,canopy)).mkdir(parents=True, exist_ok=True)

        with open("{}/{}/pairFeatures.csv".format(origDir,canopy),'r') as origFile:
            with open("{}/{}/pairFeatures.csv".format(newDir,canopy),'w') as newFile:
                for line in origFile:
                    lineV = line.strip().split(",")
                    if lineV[-1] == "+":
                        lineV[-1] = "1"
                    elif lineV[-1] == "-":
                        lineV[-1] = "0"
                    else:
                        raise Exception("Invalid last token ..",lineV)

                    processedLine = ",".join([str(v) for v in lineV[1:]]) # Exclude doc number

                    newFile.write(processedLine +"\n")

        with open("{}/{}/gtClusters.tsv".format(origDir,canopy),'r') as origFile:
            with open("{}/{}/gtClusters.tsv".format(newDir,canopy),'w') as newFile:
                for line in origFile:
                    newFile.write(line)

if __name__ == '__main__':
    origDir = "../data/NP_Coref_withDocNum"
    newDir  = "../data/NP_Coref"

    # "This was to remove docNum present in front of each line in pairFeatures.tsv
    fixDataFormat(origDir=origDir, newDir=newDir)