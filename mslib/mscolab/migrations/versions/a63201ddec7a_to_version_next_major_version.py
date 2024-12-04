"""To version <next-major-version>

Revision ID: a63201ddec7a
Revises: 922e4d9c94e2
Create Date: 2024-11-28 14:11:53.302308

"""
from alembic import op
import sqlalchemy as sa
import mslib.mscolab.custom_migration_types as cu


# revision identifiers, used by Alembic.
revision = 'a63201ddec7a'
down_revision = '922e4d9c94e2'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.add_column(sa.Column('fullname', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('nickname', sa.String(length=255), nullable=True))

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.drop_column('nickname')
        batch_op.drop_column('fullname')

    # ### end Alembic commands ###
