from sqlalchemy import Table, Column, Integer, String, ForeignKey


def create_imported_files_table(metadata):
    imported_files_table = Table("imported_files", metadata,
                                 Column("id", Integer, primary_key=True),
                                 Column("file_name", String, nullable=False),
                                 Column("scan_id", Integer, ForeignKey("scans.id"))
                                 )
    return imported_files_table
