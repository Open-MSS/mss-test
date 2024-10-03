"""To version 9.0.0

Revision ID: c171019fe3ee
Revises: 92eaba86a92e
Create Date: 2024-07-08 15:49:08.277483

"""
from alembic import op
import sqlalchemy as sa
import mslib.mscolab.custom_migration_types as cu


# revision identifiers, used by Alembic.
revision = 'c171019fe3ee'
down_revision = '92eaba86a92e'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.add_column(sa.Column('authentication_backend', sa.String(length=255), nullable=False, default='local'))
        batch_op.drop_constraint('uq_users_password', type_='unique')

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.create_unique_constraint('uq_users_password', ['password'])
        batch_op.drop_column('authentication_backend')

    # ### end Alembic commands ###
