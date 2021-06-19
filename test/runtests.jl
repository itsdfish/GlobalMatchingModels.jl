using SafeTestsets

files = [
    "MINERVA_tests.jl",
    "REM_tests.jl"
]

for file in files 
    include(file)
end
