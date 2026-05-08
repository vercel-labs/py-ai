import pydantic


class ProviderMetadata(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        frozen=True,
        populate_by_name=True,
        extra="allow",
    )
