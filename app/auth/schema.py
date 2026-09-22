from pydantic import BaseModel, EmailStr, model_validator

class SignupRequest(BaseModel):
    name: str
    email: EmailStr
    mobile: str
    password: str

class LoginRequest(BaseModel):
    email: EmailStr | None = None
    mobile: str | None = None
    password: str

    @model_validator(mode="after")
    def check_identifier_present(self):
        if not self.mobile and not self.email:
            raise ValueError("Either email or mobile must be provided")
        else:
            return self
