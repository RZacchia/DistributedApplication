namespace BookRent.Identity.Api;

internal static class IdentityEndpoints
{
    internal static IResult Login(HttpContext context)
    {
        return Results.Ok();
    }
    
    internal static IResult Register(HttpContext context)
    {
        return Results.Ok();
    }
    
    internal static IResult GetUserRole(HttpContext context)
    {
        return Results.Ok();
    }
}