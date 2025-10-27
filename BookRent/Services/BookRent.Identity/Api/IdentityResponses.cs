namespace BookRent.Identity.Api;

record TokenResponse(string AccessToken, string TokenType, int ExpiresIn);
