namespace BookRent.User.Api;

internal static class UserDataEndpoints
{
    internal static IResult GetUser(HttpRequest request)
    {
        
        return TypedResults.Ok("success");
    }
}