def filter_definitions(definitions):
    return list(
        filter(lambda x: not x["definition"].startswith("variant of"), definitions)
    )
