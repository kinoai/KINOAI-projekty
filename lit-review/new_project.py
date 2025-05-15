import os

template = open("template.md", "r").read()

name = input("Nazwa projektu: ")
maintainer = input("Główny ogarniacz: ")
goal = input("Cel projektu: ")
description = input("Opis: ")

print("Bibliografia [pusty wpis by zakończyć]")
refs = []
while True:
    ref = input(f"{len(refs) + 1}. ")
    if len(ref) == 0:
        break
    refs.append(f"{len(refs) + 1}. " + ref)

refs = "\\\n".join(refs)



project_name = name.replace(" ", "-").lower()

template = template.format(name=name, goal=goal, bib=refs, description=description, maintainer=maintainer)
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, project_name)
os.makedirs(final_directory)
os.chdir(final_directory)

open("README.md", "w").write(template)



